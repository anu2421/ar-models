"""
Week 2: fine-tune a ProGen2 model on the real AMP corpus. Works for the baseline (small)
or challenger (medium) via --model-name / --tag — outputs go to separate, non-colliding
paths so a challenger run never overwrites the baseline's results.

Training set = split in {"core_train_only", "train"} → 26,699 sequences (per team
decision, Sept 2026 — train alone was judged too small for a 151M-param model).
validation and test stay held out for evaluation, exactly as agreed.

Usage:
    python scripts/05_finetune.py --views-dir /content/AMP/data/processed/views \
        --epochs 3 --batch-size 8 --lr 5e-5

    # challenger run:
    python scripts/05_finetune.py --views-dir /content/AMP/data/processed/views \
        --model-name hugohrban/progen2-medium --tag medium \
        --epochs 2 --batch-size 4 --lr 5e-5

Deliverables (per the guide's Step 4 + Day-by-day Week 2 requirements):
    outputs/checkpoints/<tag or 'default'>/epoch_N/   — model + tokenizer at each epoch
    docs/finetune_manifest<_tag>.json                 — checkpoint, batch size, lr, seed, data version
    docs/finetune_log<_tag>.csv                       — loss per step, for the comparison report
"""

import os
import sys
import json
import argparse
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common.data_view import load_ar_view
from common.model_utils import load_model_and_tokenizer, START_TOKEN, END_TOKEN, MODEL_NAME

TRAIN_SPLITS = {"core_train_only", "train"}
MAX_SEQ_LEN = 52  # 50 residues + start + end tokens


class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_len=MAX_SEQ_LEN):
        self.examples = []
        for seq in sequences:
            ids = tokenizer.encode(START_TOKEN + seq + END_TOKEN).ids
            if len(ids) > max_len:
                ids = ids[:max_len]
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


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
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    vocab_size = logits.size(-1)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    flat_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    flat_loss = flat_loss.view(shift_labels.shape)

    masked_loss = (flat_loss * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)
    return masked_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-dir", required=True)
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
                         help="Optional hard cap on training steps, for a quick test run.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt_subdir = args.tag if args.tag else "default"

    df = load_ar_view(args.views_dir)
    data_version = df["data_version"].iloc[0] if "data_version" in df.columns else "unknown"

    train_df = df[df["split"].isin(TRAIN_SPLITS)]
    val_df = df[df["split"] == "validation"]
    print(f"Training on {len(train_df)} sequences, validating on {len(val_df)}.")

    model, tokenizer, device = load_model_and_tokenizer(model_name=args.model_name)
    pad_id = tokenizer.encode(END_TOKEN).ids[0]  # reuse end-token id as pad filler

    train_ds = SequenceDataset(train_df["sequence"].tolist(), tokenizer)
    val_ds = SequenceDataset(val_df["sequence"].tolist(), tokenizer)
    collate = make_collate_fn(pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_path = os.path.join(ROOT, "docs", f"finetune_log{suffix}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "split", "loss"])

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for input_ids, attention_mask in train_loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            loss = compute_loss(model, input_ids, attention_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 20 == 0:
                print(f"epoch {epoch} step {global_step} train_loss {loss.item():.4f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, global_step, "train", loss.item()])

            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

        # end-of-epoch validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for input_ids, attention_mask in val_loader:
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                val_losses.append(compute_loss(model, input_ids, attention_mask).item())
        mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        print(f"epoch {epoch} done — mean val_loss {mean_val_loss:.4f}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, global_step, "val", mean_val_loss])
        model.train()

        ckpt_dir = os.path.join(ROOT, "outputs", "checkpoints", ckpt_subdir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save(os.path.join(ckpt_dir, "tokenizer.json"))
        print(f"Saved checkpoint: {ckpt_dir}")

    manifest = {
        "base_model": args.model_name,
        "data_version": str(data_version),
        "train_splits_used": sorted(TRAIN_SPLITS),
        "n_train_sequences": len(train_df),
        "n_val_sequences": len(val_df),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "max_seq_len": MAX_SEQ_LEN,
    }
    manifest_path = os.path.join(ROOT, "docs", f"finetune_manifest{suffix}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {manifest_path}")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()