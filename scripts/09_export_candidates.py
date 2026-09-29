"""
Step 7 deliverable: ar_candidates.parquet + generation_manifest.json.

"Remove generation tokens, preserve raw output, attach run metadata and send candidates to
the shared evaluator. Do not silently use your own final ranking."

This is the handover file. The Shared Evaluator role merges four domains' pools
automatically, so the column contract matters more than anything else here: every row
carries its own IDs, domain, model, run and generation metadata, so a merged table can
always be traced back to the run that produced it.

Note on `raw_score`: the evaluator owns scoring. We export our own raw model log-likelihood
as *evidence*, clearly named, and we do NOT rank or pre-filter by it — the guide is explicit
that a generator must not silently apply its own final ranking. Invalid sequences are kept
in the file too, flagged rather than dropped, so the evaluator sees the true pool.

Usage:
    python scripts/09_export_candidates.py \
        --candidates outputs/finetuned_candidates.csv \
        --views-dir /content/AMP/data/processed/views \
        --checkpoint outputs/checkpoints/default/epoch_2 \
        --run-id ar_ft_run1

    # merge several pools into one export
    python scripts/09_export_candidates.py \
        --candidates outputs/sampling_candidates.csv outputs/finetuned_candidates.csv \
        --views-dir ... --run-id ar_combined

Writes:
    outputs/ar_candidates.parquet     — the handover file (csv fallback if pyarrow missing)
    outputs/generation_manifest.json  — how it was produced
    outputs/SHA256SUMS                — checksums for the reproducibility report
"""

import os
import sys
import json
import time
import hashlib
import argparse
import platform
import subprocess

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common.data_view import load_ar_view, resolve_data_version
from common.validity import is_valid_sequence, summarize_validity, MIN_LENGTH, MAX_LENGTH
from common.novelty import measure_overlap, measure_internal_redundancy

DOMAIN = "ar"
TRAIN_SPLITS = {"core_train_only", "train"}

# The column contract. Keep these names stable — the evaluator merges on them.
CANDIDATE_COLUMNS = [
    "sequence_id",        # globally unique, domain-prefixed
    "sequence",           # cleaned, uppercase, no start/end tokens
    "domain",             # always "ar" for this repo
    "model_name",         # base model or checkpoint that produced it
    "run_id",             # which generation run
    "length",
    "is_valid",           # against the shared challenge rules
    "invalid_reason",
    "raw_score",          # our own evidence, NOT a ranking the evaluator should trust
    "raw_score_type",
    "temperature",
    "top_p",
    "seed",
    "exact_training_match",
    "near_training_match",
    "nearest_edit_distance",
    "data_version",       # which frozen DE view this run used
    "generated_at",
]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sequence_uid(seq: str, run_id: str, idx: int) -> str:
    """Stable, collision-resistant, and readable enough to grep for in a merged table."""
    digest = hashlib.sha1(f"{run_id}|{idx}|{seq}".encode()).hexdigest()[:10]
    return f"{DOMAIN}_{run_id}_{idx:06d}_{digest}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, nargs="+",
                        help="One or more candidate CSVs to merge into the export.")
    parser.add_argument("--views-dir", required=True)
    parser.add_argument("--view-path", default=None)
    parser.add_argument("--run-id", required=True,
                        help="Short identifier for this generation run, e.g. ar_ft_run1.")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint used, recorded in the manifest.")
    parser.add_argument("--model-name", default=None,
                        help="Override the model name recorded per row.")
    parser.add_argument("--out", default="ar_candidates",
                        help="Output basename inside outputs/.")
    parser.add_argument("--max-edits", type=int, default=1)
    parser.add_argument("--keep-invalid", action="store_true", default=True,
                        help="Keep invalid sequences in the export, flagged. On by default — "
                             "the guide warns against aggressive early filtering.")
    parser.add_argument("--drop-invalid", dest="keep_invalid", action="store_false",
                        help="Drop invalid rows instead. Use only if the evaluator asks.")
    parser.add_argument("--dedupe", action="store_true",
                        help="Drop exact duplicate sequences within this export.")
    args = parser.parse_args()

    # ---- gather candidate rows ----------------------------------------------
    frames = []
    for path in args.candidates:
        df = pd.read_csv(path)
        if "sequence" not in df.columns:
            raise SystemExit(f"{path} has no 'sequence' column.")
        df["_source_file"] = os.path.basename(path)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["sequence"] = raw["sequence"].astype(str).str.strip().str.upper()
    raw = raw[raw["sequence"].str.len() > 0]

    n_before = len(raw)
    if args.dedupe:
        raw = raw.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
        print(f"Deduplicated: {n_before} -> {len(raw)} rows")

    print(f"Exporting {len(raw)} candidates from {len(args.candidates)} file(s).")

    # ---- reference corpus for the overlap columns ---------------------------
    view = load_ar_view(args.views_dir, path=args.view_path)
    data_version = resolve_data_version(view)
    training = [
        str(s).strip().upper()
        for s in view[view["split"].isin(TRAIN_SPLITS)]["sequence"].dropna()
    ]
    print(f"Measuring overlap against {len(training)} training sequences...")
    overlap = measure_overlap(raw["sequence"].tolist(), training, max_edits=args.max_edits)
    ov_rows = overlap["per_candidate"]

    # ---- build the contract table -------------------------------------------
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    model_name = args.model_name or args.checkpoint or "unknown"

    records = []
    for i, (_, r) in enumerate(raw.iterrows()):
        seq = r["sequence"]
        valid, reason = is_valid_sequence(seq)
        ov = ov_rows[i]
        records.append({
            "sequence_id": sequence_uid(seq, args.run_id, i),
            "sequence": seq,
            "domain": DOMAIN,
            "model_name": r.get("model_name", r.get("checkpoint", model_name)),
            "run_id": args.run_id,
            "length": len(seq),
            "is_valid": valid,
            "invalid_reason": None if valid else reason,
            "raw_score": r.get("raw_score", None),
            "raw_score_type": "none_exported",
            "temperature": r.get("temperature", None),
            "top_p": r.get("top_p", None),
            "seed": r.get("seed", None),
            "exact_training_match": ov["exact_training_match"],
            "near_training_match": ov["near_training_match"],
            "nearest_edit_distance": ov["nearest_edit_distance"],
            "data_version": data_version,
            "generated_at": generated_at,
        })

    out = pd.DataFrame(records, columns=CANDIDATE_COLUMNS)

    n_invalid = int((~out["is_valid"]).sum())
    if not args.keep_invalid:
        out = out[out["is_valid"]].reset_index(drop=True)
        print(f"Dropped {n_invalid} invalid rows (--drop-invalid).")
    elif n_invalid:
        print(f"Keeping {n_invalid} invalid rows, flagged (is_valid=False).")

    assert out["sequence_id"].is_unique, "sequence_id collision — this must never happen"

    # ---- write ---------------------------------------------------------------
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    parquet_path = os.path.join(ROOT, "outputs", f"{args.out}.parquet")
    written = []
    try:
        out.to_parquet(parquet_path, index=False)
        written.append(parquet_path)
    except Exception as e:
        print(f"Parquet write failed ({e}); falling back to CSV. Install pyarrow for parquet.")
        parquet_path = os.path.join(ROOT, "outputs", f"{args.out}.csv")
        out.to_csv(parquet_path, index=False)
        written.append(parquet_path)

    # ---- manifest ------------------------------------------------------------
    validity = summarize_validity(out["sequence"].tolist())
    redundancy = measure_internal_redundancy(out["sequence"].tolist(), max_edits=2)

    manifest = {
        "run_id": args.run_id,
        "domain": DOMAIN,
        "generated_at": generated_at,
        "export_file": os.path.basename(parquet_path),
        "n_candidates": len(out),
        "n_invalid_kept": int((~out["is_valid"]).sum()),
        "source_candidate_files": [os.path.basename(p) for p in args.candidates],
        "deduplicated": args.dedupe,
        "model": {
            "model_name": model_name,
            "checkpoint": args.checkpoint,
        },
        "data": {
            "data_version": data_version,
            "view_source_path": view.attrs.get("source_path"),
            "train_splits_used": sorted(TRAIN_SPLITS),
            "n_training_sequences": len(training),
        },
        "validity_rules": {
            "alphabet": "20 standard amino acids",
            "min_length": MIN_LENGTH,
            "max_length": MAX_LENGTH,
        },
        "metrics": {
            "validity": validity,
            "training_overlap": overlap["summary"],
            "internal_redundancy": {
                k: v for k, v in redundancy.items() if k != "top_clusters"
            },
        },
        "scoring_note": (
            "No ranking applied. raw_score is empty unless a scorer was run separately. "
            "Final scoring and selection belong to the Shared Evaluator role."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": git_commit(),
        },
        "column_contract": CANDIDATE_COLUMNS,
    }
    manifest_path = os.path.join(ROOT, "outputs", "generation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    written.append(manifest_path)

    # ---- checksums -----------------------------------------------------------
    sums_path = os.path.join(ROOT, "outputs", "SHA256SUMS")
    with open(sums_path, "w") as f:
        for p in written:
            f.write(f"{sha256_file(p)}  {os.path.basename(p)}\n")

    print("\n--- Export summary ---")
    print(json.dumps({
        "n_candidates": len(out),
        "validity_rate": validity["validity_rate"],
        "exact_match_rate": overlap["summary"]["exact_match_rate"],
        "near_match_rate": overlap["summary"]["near_match_rate"],
        "novelty_rate": overlap["summary"]["novelty_rate"],
        "effective_diversity": redundancy["effective_diversity"],
        "data_version": data_version,
    }, indent=2))
    for p in written + [sums_path]:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
