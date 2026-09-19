"""
Generate a small pool from a fine-tuned checkpoint and compare it against the pretrained
baseline's smoke test (which had mean_length 50.0 — nothing stopped naturally). This
tells us whether fine-tuning actually taught the model to produce properly-terminated
short peptides.

Usage:
    python scripts/06_generate_from_checkpoint.py --checkpoint outputs/checkpoints/epoch_1 --n 100
"""

import os
import sys
import json
import argparse

import torch
import pandas as pd
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common.model_utils import generate_sequence, MODEL_NAME
from common.validity import is_valid_sequence, summarize_validity


def load_finetuned(checkpoint_dir: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model.to(device)
    model.eval()
    # Tokenizer is unchanged by fine-tuning, so load the original rather than from checkpoint
    tokenizer = Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.no_padding()
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a saved epoch_N checkpoint dir")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model, tokenizer, device = load_finetuned(args.checkpoint)

    records = []
    for i in range(args.n):
        seed = args.seed + i
        seq = generate_sequence(model, tokenizer, device, seed=seed)
        valid, reason = is_valid_sequence(seq)
        records.append({
            "run_id": f"ft_{i:03d}",
            "sequence": seq,
            "length": len(seq),
            "valid": valid,
            "reason": reason,
            "checkpoint": args.checkpoint,
            "seed": seed,
        })
        if (i + 1) % 10 == 0:
            print(f"  generated {i + 1}/{args.n}")

    df = pd.DataFrame(records)
    out_csv = os.path.join(ROOT, "outputs", "finetuned_candidates.csv")
    df.to_csv(out_csv, index=False)

    metrics = summarize_validity(df["sequence"].tolist())
    metrics["mean_length"] = round(float(df["length"].mean()), 2)
    metrics["checkpoint"] = args.checkpoint

    out_json = os.path.join(ROOT, "outputs", "finetuned_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Fine-tuned generation summary ---")
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")
    print("\nCompare mean_length above against the pretrained baseline's 50.0 from "
          "outputs/smoke_metrics.json — a drop toward realistic AMP lengths (most known "
          "AMPs are well under 50 residues) means fine-tuning worked as intended.")


if __name__ == "__main__":
    main()