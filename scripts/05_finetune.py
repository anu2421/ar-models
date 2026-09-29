"""
Week 2: fine-tune a ProGen2 model on the real AMP corpus. Works for the baseline (small)
or challenger (medium) via --model-name / --tag — outputs go to separate, non-colliding
paths so a challenger run never overwrites the baseline's results.

Training set = split in {"core_train_only", "train"} -> 26,699 sequences (per team
decision, Sept 2026 — train alone was judged too small for a 151M-param model).
validation and test stay held out for evaluation, exactly as agreed.

Usage:
    python scripts/05_finetune.py --views-dir /content/AMP/data/processed/views \
        --epochs 3 --batch-size 8 --lr 5e-5

    # challenger run, memory-constrained GPU:
    python scripts/05_finetune.py --views-dir /content/AMP/data/processed/views \
        --model-name hugohrban/progen2-medium --tag medium \
        --epochs 2 --batch-size 4 --lr 5e-5 --amp

    # 30-step smoke run that still writes a checkpoint and a manifest:
    python scripts/05_finetune.py --views-dir ... --max-steps 30

Deliverables (per the guide's Step 4 + Day-by-day Week 2 requirements):
    outputs/checkpoints/<tag or 'default'>/epoch_N/   — model + tokenizer at each epoch
    docs/finetune_manifest<_tag>.json                 — checkpoint, batch size, lr, seed, data version
    docs/finetune_log<_tag>.csv                       — loss per step, for the comparison report

Precision note (supersedes the earlier fp16 -> bf16 commit):
    Plain fp16 master weights produced NaN loss. The fix used at the time was to load the
    master weights in bf16, which stops the NaN but trains AdamW on bf16 weights — at
    lr 5e-5 the updates are near the edge of bf16's 8-bit mantissa, so small updates get
    rounded away. `--amp` is the correct fix: fp32 master weights, autocast forward/backward,
    and a GradScaler. `--bf16-weights` keeps the old behaviour for reproducing earlier runs.
"""

import os
import sys
import json
import argparse
import csv
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common.data_view import load_ar_view, check_schema, resolve_data_version
from common.model_utils import load_model_and_tokenizer, START_TOKEN, END_TOKEN, MODEL_NAME
from common.validity import MAX_LENGTH

TRAIN_SPLITS = {"core_train_only", "train"}
VAL_SPLITS = {"validation", "val"}
MAX_SEQ_LEN = MAX_LENGTH + 2  # 50 residues + start + end tokens


class SequenceDataset(Dataset):
    """
    Tokenised sequences as [START] + residues + [END].

    Over-length rows are DROPPED, not truncated. Truncating at MAX_SEQ_LEN cuts the END
    token off the row, which teaches the model a sequence that never terminates — the
    model then relies entirely on the sampler's max_len to stop, and validity collapses
    whenever that guard is relaxed. The challenge's own rules cap valid peptides at 50
    residues anyway, so a longer row is out-of-spec input, not data to salvage.
    """

    def __init__(self, sequences, tokenizer, max_len=MAX_SEQ_LEN):
        self.examples = []
        self.n_dropped_too_long = 0
        self.n_dropped_empty = 0

        for seq in sequences:
            seq = str(seq).strip().upper()
            if not seq:
                self.n_dropped_empty += 1
                continue
            ids = tokenizer.encode(START_TOKEN + seq + END_TOKEN).ids
            if len(ids) > max_len:
                self.n_dropped_too_long += 1
                continue
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def report(self, label: str):
        print(
            f"  {label}: kept {len(self.examples)}"
            f", dropped {self.n_dropped_too_long} over {MAX_SEQ_LEN} tokens"
            f", dropped {self.n_dropped_empty} empty"
        )
        return {
            "kept": len(self.examples),
            "dropped_too_long": self.n_dropped_too_long,
            "dropped_empty": self.n_dropped_empty,
        }


def make_collate_fn(pad_id):
    def collate(batch):
        max_len = max(len(ids) for ids in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return input_ids, attention_mask

    return collate


def compute_loss(model, input_ids, attention_mask):
    """
    Manual causal-LM loss (shift-by-one, ignore padding) rather than relying on the
    model's internal `labels` handling. vocab_size is read from the model's own output
    shape (not the tokenizer's reported vocab size) since they can differ.

    Verified: padding never enters the loss, and the real END token IS scored in every
    row, so the model learns when to stop.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    vocab_size = logits.size(-1)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    flat_loss = loss_fct(shift_logits.view(-1, vocab_size).float(), shift_labels.view(-1))
    flat_loss = flat_loss.view(shift_labels.shape)

    masked_loss = (flat_loss * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)
    return masked_loss


def lr_lambda_factory(warmup_steps: int, total_steps: int):
    """Linear warmup then linear decay. Warmup is what prevents the step-1 loss spike."""
    def fn(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    return fn


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    losses = []
    for input_ids, attention_mask in loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            losses.append(compute_loss(model, input_ids, attention_mask).item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def save_checkpoint(model, tokenizer, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save(os.path.join(ckpt_dir, "tokenizer.json"))
    print(f"Saved checkpoint: {ckpt_dir}")
    return ckpt_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-dir", required=True)
    parser.add_argument("--view-path", default=None,
                        help="Exact AR view file, bypassing auto-detection. Prefer this "
                             "once Data Engineering tells you the real filename.")
    parser.add_argument("--model-name", default=MODEL_NAME,
                        help="HuggingFace model repo, e.g. hugohrban/progen2-medium")
    parser.add_argument("--tag", default="",
                        help="Subfolder/suffix for outputs, e.g. 'medium' keeps challenger "
                             "runs separate from the baseline's checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Hard cap on training steps for a quick test run. A capped "
                             "run still validates, saves a checkpoint and writes a manifest.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="Fraction of total steps spent warming the LR up from 0.")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max global grad norm. 0 disables clipping.")
    parser.add_argument("--amp", action="store_true",
                        help="Mixed precision: fp32 master weights + bf16 autocast. This is "
                             "the recommended way to fit progen2-medium on a free-tier GPU.")
    parser.add_argument("--bf16-weights", action="store_true",
                        help="Load master weights in bf16 (the older, less numerically sound "
                             "approach). Kept only to reproduce earlier runs.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt_subdir = args.tag if args.tag else "default"

    # ---- data ----------------------------------------------------------------
    df = load_ar_view(args.views_dir, path=args.view_path)
    schema = check_schema(df)
    if schema["missing_required"]:
        raise SystemExit(
            f"AR view is missing required columns: {schema['missing_required']}.\n"
            "Go back to Data Engineering before training — a manifest without data_version "
            "is not traceable and will not satisfy the completion criteria."
        )
    data_version = resolve_data_version(df)

    train_df = df[df["split"].isin(TRAIN_SPLITS)]
    val_df = df[df["split"].isin(VAL_SPLITS)]
    if len(train_df) == 0:
        raise SystemExit(
            f"No rows matched the training splits {sorted(TRAIN_SPLITS)}. "
            f"Splits actually present: {sorted(df['split'].unique().tolist())}"
        )
    print(f"data_version: {data_version}")
    print(f"Training on {len(train_df)} rows, validating on {len(val_df)} rows.")

    use_amp = args.amp and torch.cuda.is_available()
    if args.amp and not torch.cuda.is_available():
        print("WARNING: --amp requested but no CUDA device — running fp32 on CPU instead.")

    model, tokenizer, device = load_model_and_tokenizer(
        model_name=args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16_weights else None,
    )
    pad_id = tokenizer.encode(END_TOKEN).ids[0]  # reuse end-token id as pad filler

    train_ds = SequenceDataset(train_df["sequence"].tolist(), tokenizer)
    val_ds = SequenceDataset(val_df["sequence"].tolist(), tokenizer)
    print("Dataset build:")
    train_stats = train_ds.report("train")
    val_stats = val_ds.report("val")
    if len(train_ds) == 0:
        raise SystemExit("Every training row was dropped. Check the view's sequence column.")

    collate = make_collate_fn(pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # ---- optimisation --------------------------------------------------------
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    planned_steps = steps_per_epoch * args.epochs
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(warmup_steps, total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"{total_steps} total steps ({steps_per_epoch}/epoch), {warmup_steps} warmup steps")

    log_path = os.path.join(ROOT, "docs", f"finetune_log{suffix}.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "split", "loss", "lr", "grad_norm"])

    def log(row):
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # ---- train ---------------------------------------------------------------
    model.train()
    global_step = 0
    stopped_early = False
    saved_checkpoints = []
    epochs_completed = 0
    val_history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        for input_ids, attention_mask in train_loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = compute_loss(model, input_ids, attention_mask)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                grad_norm = float(grad_norm)
            else:
                grad_norm = float("nan")

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            cur_lr = optimizer.param_groups[0]["lr"]
            log([epoch, global_step, "train", loss.item(), cur_lr, grad_norm])

            if not math.isfinite(loss.item()):
                raise SystemExit(
                    f"Loss became {loss.item()} at step {global_step}. Stopping rather than "
                    "writing a poisoned checkpoint. Try --amp instead of --bf16-weights, "
                    "or lower --lr."
                )
            if global_step % 20 == 0:
                print(f"epoch {epoch} step {global_step}/{total_steps} "
                      f"loss {loss.item():.4f} lr {cur_lr:.2e} gnorm {grad_norm:.2f}")

            if args.max_steps and global_step >= args.max_steps:
                stopped_early = True
                break

        # Validation + checkpoint run whether the epoch finished naturally or --max-steps
        # cut it short. (Previously an early break skipped both, so a capped run produced
        # no checkpoint and no validation number at all.)
        mean_val_loss = evaluate(model, val_loader, device, use_amp)
        val_history.append({"epoch": epoch, "step": global_step, "val_loss": mean_val_loss})
        print(f"epoch {epoch} — mean val_loss {mean_val_loss:.4f}")
        log([epoch, global_step, "val", mean_val_loss, optimizer.param_groups[0]["lr"], ""])

        ckpt_dir = os.path.join(ROOT, "outputs", "checkpoints", ckpt_subdir, f"epoch_{epoch}")
        saved_checkpoints.append(save_checkpoint(model, tokenizer, ckpt_dir))
        epochs_completed = epoch + 1

        if stopped_early:
            print(f"Stopping: --max-steps {args.max_steps} reached.")
            break

    elapsed = time.time() - t0

    # ---- manifest ------------------------------------------------------------
    manifest = {
        "base_model": args.model_name,
        "data_version": str(data_version),
        "view_source_path": schema["source_path"],
        "applied_column_aliases": schema["applied_aliases"],
        "train_splits_used": sorted(TRAIN_SPLITS),
        "val_splits_used": sorted(VAL_SPLITS),
        "n_train_rows_in_view": len(train_df),
        "n_val_rows_in_view": len(val_df),
        "train_dataset": train_stats,
        "val_dataset": val_stats,
        "epochs_requested": args.epochs,
        "epochs_completed": epochs_completed,
        "steps_completed": global_step,
        "steps_planned": total_steps,
        "stopped_early_by_max_steps": stopped_early,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "max_seq_len": MAX_SEQ_LEN,
        "precision": (
            "amp_bf16_autocast_fp32_master" if use_amp
            else "bf16_master_weights" if args.bf16_weights
            else "fp32"
        ),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_mem_gb": (
            round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if torch.cuda.is_available() else None
        ),
        "wall_clock_seconds": round(elapsed, 1),
        "val_history": val_history,
        "checkpoints": [os.path.relpath(c, ROOT) for c in saved_checkpoints],
        "torch_version": torch.__version__,
    }
    manifest_path = os.path.join(ROOT, "docs", f"finetune_manifest{suffix}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {manifest_path}")
    print(f"Wrote {log_path}")
    print(f"Total wall clock: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
