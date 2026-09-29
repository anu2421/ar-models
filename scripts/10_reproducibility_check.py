"""
Step 8 deliverable: reproducibility_check.md with hashes.

"Run the documented command twice with the fixed seed and confirm identical output for the
small reference job."

This runs a small generation job twice in separate subprocesses, hashes both outputs, and
reports whether they are byte-identical. Two separate processes — not two loops inside one
process — because that is what actually catches hidden global state, and it is what a
reviewer would do.

Usage:
    # pretrained baseline
    python scripts/10_reproducibility_check.py --n 25

    # a fine-tuned checkpoint
    python scripts/10_reproducibility_check.py \
        --checkpoint outputs/checkpoints/default/epoch_2 --n 25

Writes:
    docs/reproducibility_check.md
    outputs/repro_run_a.csv, outputs/repro_run_b.csv

If the two runs differ, the report says WHERE they diverge (first differing index), which
is usually far more useful than just "not reproducible".
"""

import os
import sys
import json
import hashlib
import argparse
import platform
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# This is the worker body: it runs in a fresh subprocess, generates N sequences with a
# fixed seed, and prints them one per line. Kept inline so there is no extra file to
# keep in sync with the main sampling code.
WORKER = r"""
import os, sys, json
ROOT = sys.argv[1]
sys.path.insert(0, ROOT)
import torch
from common.model_utils import load_model_and_tokenizer, generate_sequence, MODEL_NAME

checkpoint = sys.argv[2] or None
n = int(sys.argv[3]); seed = int(sys.argv[4])
temperature = float(sys.argv[5]); top_p = float(sys.argv[6])

if checkpoint:
    from transformers import AutoModelForCausalLM
    from tokenizers import Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
    model.to(device); model.eval()
    tp = os.path.join(checkpoint, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tp) if os.path.exists(tp) else Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.no_padding()
else:
    model, tokenizer, device = load_model_and_tokenizer()

out = []
for i in range(n):
    out.append(generate_sequence(model, tokenizer, device,
                                 temperature=temperature, top_p=top_p, seed=seed + i))
print("__BEGIN__")
for s in out:
    print(s)
print("__END__")
print(json.dumps({"device": device, "torch": torch.__version__,
                  "cuda": torch.cuda.is_available()}))
"""


def run_once(label, checkpoint, n, seed, temperature, top_p):
    worker_path = os.path.join(ROOT, "outputs", "_repro_worker.py")
    os.makedirs(os.path.dirname(worker_path), exist_ok=True)
    with open(worker_path, "w") as f:
        f.write(WORKER)

    print(f"Running {label} in a fresh subprocess...")
    proc = subprocess.run(
        [sys.executable, worker_path, ROOT, checkpoint or "", str(n),
         str(seed), str(temperature), str(top_p)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-4000:])
        raise SystemExit(f"{label} failed with exit code {proc.returncode}")

    lines = proc.stdout.splitlines()
    try:
        b = lines.index("__BEGIN__")
        e = lines.index("__END__")
    except ValueError:
        print(proc.stdout[-2000:])
        raise SystemExit(f"{label} produced no parsable output.")

    seqs = lines[b + 1:e]
    env = {}
    if e + 1 < len(lines):
        try:
            env = json.loads(lines[e + 1])
        except Exception:
            pass
    return seqs, env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n", type=int, default=25,
                        help="Sequences in the small reference job.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    a, env_a = run_once("run A", args.checkpoint, args.n, args.seed, args.temperature, args.top_p)
    b, env_b = run_once("run B", args.checkpoint, args.n, args.seed, args.temperature, args.top_p)

    for label, seqs in (("a", a), ("b", b)):
        p = os.path.join(ROOT, "outputs", f"repro_run_{label}.csv")
        with open(p, "w") as f:
            f.write("index,sequence\n")
            for i, s in enumerate(seqs):
                f.write(f"{i},{s}\n")

    hash_a = hashlib.sha256("\n".join(a).encode()).hexdigest()
    hash_b = hashlib.sha256("\n".join(b).encode()).hexdigest()
    identical = a == b

    first_diff = None
    if not identical:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                first_diff = {"index": i, "run_a": x, "run_b": y}
                break
        if first_diff is None:
            first_diff = {"index": min(len(a), len(b)),
                          "note": f"different lengths: A={len(a)} B={len(b)}"}

    worker_path = os.path.join(ROOT, "outputs", "_repro_worker.py")
    if os.path.exists(worker_path):
        os.remove(worker_path)

    lines = [
        "# Reproducibility check",
        "",
        "Two independent subprocesses, same seed, same settings, compared byte for byte.",
        "",
        "## Job",
        "",
        f"- Model: `{args.checkpoint or 'hugohrban/progen2-small (pretrained)'}`",
        f"- Sequences: {args.n}",
        f"- Base seed: {args.seed} (sequence *i* uses seed {args.seed}+*i*)",
        f"- temperature {args.temperature}, top_p {args.top_p}",
        f"- Device: `{env_a.get('device', 'unknown')}`, torch `{env_a.get('torch', '?')}`",
        f"- Python {platform.python_version()} on {platform.platform()}",
        "",
        "## Result",
        "",
        f"**{'IDENTICAL — reproducible' if identical else 'DIVERGED — not reproducible'}**",
        "",
        "| Run | SHA256 of concatenated output |",
        "| --- | --- |",
        f"| A | `{hash_a}` |",
        f"| B | `{hash_b}` |",
        "",
        "## Command to reproduce",
        "",
        "```bash",
        f"python scripts/10_reproducibility_check.py --n {args.n} --seed {args.seed} "
        f"--temperature {args.temperature} --top-p {args.top_p}"
        + (f" \\\n    --checkpoint {args.checkpoint}" if args.checkpoint else ""),
        "```",
    ]

    if not identical:
        lines += [
            "",
            "## Divergence",
            "",
            f"First difference at index {first_diff.get('index')}:",
            "",
            f"- run A: `{first_diff.get('run_a', '-')}`",
            f"- run B: `{first_diff.get('run_b', '-')}`",
            "",
            "Likely causes, in the order worth checking:",
            "",
            "1. Non-deterministic GPU kernels. Set `torch.use_deterministic_algorithms(True)`",
            "   and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, then re-run.",
            "2. A seed set once globally rather than per sequence — `generate_sequence` calls",
            "   `torch.manual_seed(seed)` per sequence, so each index must be independent.",
            "3. Different device between runs (CPU vs GPU produce different draws).",
        ]
    else:
        lines += [
            "",
            "## What this does and does not prove",
            "",
            "It proves the sampling path is deterministic for a fixed seed on THIS device.",
            "It does not prove CPU and GPU agree with each other, nor that results hold",
            "across torch versions. Record the device and version alongside any published",
            "numbers.",
        ]

    md = os.path.join(ROOT, "docs", "reproducibility_check.md")
    os.makedirs(os.path.dirname(md), exist_ok=True)
    with open(md, "w") as f:
        f.write("\n".join(lines))

    print("\n--- Reproducibility ---")
    print(f"run A sha256: {hash_a}")
    print(f"run B sha256: {hash_b}")
    print("IDENTICAL" if identical else f"DIVERGED at index {first_diff.get('index')}")
    print(f"\nWrote {md}")

    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
