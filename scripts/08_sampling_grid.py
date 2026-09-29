"""
The sampling-grid deliverable (guide Step 5 + Week 2): sweep temperature and top-p, and
record how validity, diversity and mean length respond. This is what justifies the
sampling settings used for the final export instead of leaving them at defaults.

Usage:
    # baseline pretrained model
    python scripts/08_sampling_grid.py --n 50 --tag baseline

    # a fine-tuned checkpoint
    python scripts/08_sampling_grid.py --checkpoint outputs/checkpoints/default/epoch_2 \
        --n 50 --tag finetuned

    # custom grid
    python scripts/08_sampling_grid.py --temperatures 0.7 0.9 1.1 --top-ps 0.9 0.95 --n 100

Writes:
    outputs/sampling_comparison<_tag>.csv   — one row per (temperature, top_p) cell
    outputs/sampling_candidates<_tag>.csv   — every sequence generated, with its settings
    docs/sampling_report<_tag>.md           — the readable summary + recommended setting

Note on cost: this generates len(temperatures) * len(top_ps) * n sequences. The default
3x3 grid at --n 50 is 450 sequences. Start there; only widen the grid if two cells look
genuinely tied.
"""

import os
import sys
import json
import time
import argparse
import itertools

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common.model_utils import load_model_and_tokenizer, generate_sequence, MODEL_NAME
from common.validity import is_valid_sequence, summarize_validity
from common.novelty import measure_internal_redundancy

DEFAULT_TEMPERATURES = [0.8, 1.0, 1.2]
DEFAULT_TOP_PS = [0.85, 0.9, 0.95]


def load_from_checkpoint(checkpoint_dir: str):
    """Mirrors 06_generate_from_checkpoint.load_finetuned so the grid works on either."""
    from transformers import AutoModelForCausalLM
    from tokenizers import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model.to(device)
    model.eval()

    tok_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = Tokenizer.from_file(tok_path)
    else:
        print(f"No tokenizer.json in {checkpoint_dir}, falling back to {MODEL_NAME}.")
        tokenizer = Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.no_padding()
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Fine-tuned checkpoint dir. Omit to sweep the pretrained model.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--tag", default="")
    parser.add_argument("--n", type=int, default=50,
                        help="Sequences per grid cell.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--top-ps", type=float, nargs="+", default=DEFAULT_TOP_PS)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed. Each cell uses the same seed sequence so cells are "
                             "compared on equal footing, not on luck.")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    if args.checkpoint:
        model, tokenizer, device = load_from_checkpoint(args.checkpoint)
        model_label = args.checkpoint
    else:
        model, tokenizer, device = load_model_and_tokenizer(model_name=args.model_name)
        model_label = args.model_name

    grid = list(itertools.product(args.temperatures, args.top_ps))
    total = len(grid) * args.n
    print(f"Sweeping {len(grid)} cells x {args.n} sequences = {total} generations on {device}.")

    all_rows = []
    cell_rows = []

    for cell_i, (temp, top_p) in enumerate(grid, start=1):
        t0 = time.time()
        seqs = []
        for i in range(args.n):
            seq = generate_sequence(
                model, tokenizer, device,
                temperature=temp, top_p=top_p, seed=args.seed + i,
            )
            seqs.append(seq)
            valid, reason = is_valid_sequence(seq)
            all_rows.append({
                "sequence": seq,
                "length": len(seq),
                "valid": valid,
                "reason": reason,
                "temperature": temp,
                "top_p": top_p,
                "seed": args.seed + i,
                "model": model_label,
            })
        elapsed = time.time() - t0

        v = summarize_validity(seqs)
        red = measure_internal_redundancy(seqs, max_edits=1)
        lengths = [len(s) for s in seqs]
        cell = {
            "temperature": temp,
            "top_p": top_p,
            "n": args.n,
            "validity_rate": v["validity_rate"],
            "n_valid": v["n_valid"],
            "unique_rate": round(v["n_unique"] / args.n, 4),
            "n_exact_duplicates": v["n_duplicates"],
            "effective_diversity": red["effective_diversity"],
            "largest_family": red["largest_cluster_size"],
            "mean_length": round(sum(lengths) / len(lengths), 2),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "seconds": round(elapsed, 1),
            "sec_per_seq": round(elapsed / args.n, 3),
        }
        cell_rows.append(cell)
        print(f"  [{cell_i}/{len(grid)}] T={temp} top_p={top_p} -> "
              f"validity {cell['validity_rate']:.2f}, eff_div {cell['effective_diversity']:.2f}, "
              f"mean_len {cell['mean_length']}, {cell['sec_per_seq']}s/seq")

    cells = pd.DataFrame(cell_rows)
    cells_path = os.path.join(ROOT, "outputs", f"sampling_comparison{suffix}.csv")
    os.makedirs(os.path.dirname(cells_path), exist_ok=True)
    cells.to_csv(cells_path, index=False)

    cand_path = os.path.join(ROOT, "outputs", f"sampling_candidates{suffix}.csv")
    pd.DataFrame(all_rows).to_csv(cand_path, index=False)

    # Recommendation: among cells with validity >= 0.9 of the best validity seen, pick the
    # most diverse. Validity alone is a bad objective — a low-temperature cell can be 100%
    # valid and produce the same 5 sequences over and over.
    best_validity = cells["validity_rate"].max()
    eligible = cells[cells["validity_rate"] >= 0.9 * best_validity]
    rec = eligible.sort_values(
        ["effective_diversity", "unique_rate", "validity_rate"], ascending=False
    ).iloc[0]

    lines = [
        f"# Sampling grid — {args.tag or 'default'}",
        "",
        f"- Model: `{model_label}`",
        f"- Device: `{device}`",
        f"- {args.n} sequences per cell, base seed {args.seed} (identical seeds across cells)",
        f"- Grid: temperature {args.temperatures} x top_p {args.top_ps}",
        "",
        "## Results",
        "",
        "| T | top_p | validity | unique | eff. diversity | largest family | mean len | s/seq |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in cells.iterrows():
        lines.append(
            f"| {r['temperature']} | {r['top_p']} | {r['validity_rate']:.2f} | "
            f"{r['unique_rate']:.2f} | {r['effective_diversity']:.2f} | "
            f"{int(r['largest_family'])} | {r['mean_length']} | {r['sec_per_seq']} |"
        )

    lines += [
        "",
        "## Recommended setting",
        "",
        f"**temperature = {rec['temperature']}, top_p = {rec['top_p']}**",
        "",
        f"- validity {rec['validity_rate']:.2f}, effective diversity "
        f"{rec['effective_diversity']:.2f}, mean length {rec['mean_length']}",
        "",
        "Selection rule: among cells within 10% of the best validity rate, take the most",
        "diverse. Validity on its own is the wrong objective — a cold cell can be perfectly",
        "valid and near-identical, which scores well on validity and badly on novelty.",
        "",
        "Because the masking in `common/model_utils.py` makes non-standard residues and",
        "short sequences unsamplable, validity here is expected to sit near 1.0 across the",
        "whole grid. That is the guard working, not the model being good — diversity and the",
        "novelty numbers from `07_overlap_check.py` are what actually discriminate between",
        "these cells.",
    ]

    md_path = os.path.join(ROOT, "docs", f"sampling_report{suffix}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print("\n--- Grid ---")
    print(cells.to_string(index=False))
    print(f"\nRecommended: T={rec['temperature']} top_p={rec['top_p']}")
    print(f"\nWrote {cells_path}")
    print(f"Wrote {cand_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
