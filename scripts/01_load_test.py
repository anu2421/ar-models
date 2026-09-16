"""
Day 1-2 deliverable: prove the baseline loads on your hardware.
Writes docs/load_test.md — the guide asks for exactly this as an end-of-day deliverable.
"""

import os
import sys
import platform
import torch
import transformers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)  # makes `common` importable regardless of cwd
from common.model_utils import load_model_and_tokenizer, MODEL_NAME


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer, device = load_model_and_tokenizer()

    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

    log_lines = [
        "# Load test — ProGen2-small baseline",
        "",
        f"- Model: `{MODEL_NAME}`",
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

    out_path = os.path.join(ROOT, "docs", "load_test.md")
    with open(out_path, "w") as f:
        f.write("\n".join(log_lines))

    print("\n".join(log_lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
