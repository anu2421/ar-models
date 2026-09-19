"""
Day 1-2 deliverable: prove a model loads on your hardware.
Writes docs/load_test.md (or docs/load_test_<tag>.md if --tag is given) — the guide asks
for exactly this as an end-of-day deliverable.

Usage:
    python scripts/01_load_test.py                                          # baseline (small)
    python scripts/01_load_test.py --model-name hugohrban/progen2-medium --tag medium
"""

import os
import sys
import argparse
import platform
import torch
import transformers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)  # makes `common` importable regardless of cwd
from common.model_utils import load_model_and_tokenizer, MODEL_NAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME,
                         help="HuggingFace model repo, e.g. hugohrban/progen2-medium")
    parser.add_argument("--tag", default="",
                         help="Suffix for output filenames, e.g. 'medium' -> load_test_medium.md")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    print(f"Loading {args.model_name} ...")
    model, tokenizer, device = load_model_and_tokenizer(model_name=args.model_name)

    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

    log_lines = [
        f"# Load test — {args.model_name}",
        "",
        f"- Model: `{args.model_name}`",
        f"- Device used: `{device}`",
        f"- Parameter count: {n_params:,}",
        f"- Tokenizer vocab size: {vocab_size}",
        f"- Python: {platform.python_version()}",
        f"- torch: {torch.__version__}",
        f"- transformers: {transformers.__version__}",
        f"- CUDA available: {torch.cuda.is_available()}",
        "",
        "Load succeeded — model produced a forward pass without error.",
    ]

    # sanity forward pass on a trivial input
    test_ids = tokenizer.encode("1ACDEFG").ids
    input_tensor = torch.tensor([test_ids], device=device)
    with torch.no_grad():
        logits = model(input_tensor).logits
    log_lines.append(f"- Sanity forward pass output shape: {tuple(logits.shape)}")

    out_path = os.path.join(ROOT, "docs", f"load_test{suffix}.md")
    with open(out_path, "w") as f:
        f.write("\n".join(log_lines))

    print("\n".join(log_lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()