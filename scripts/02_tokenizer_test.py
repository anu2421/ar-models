"""
Day 3 deliverables: tokenizer_test.csv AND length_test.md.

"Feed simple sequences through tokenization and decoding. Confirm that all 20 standard
amino acids survive unchanged and that start/end tokens do not appear in exported
sequences." Plus: "Test all amino acids, start/end behavior and short-sequence generation."

The earlier version covered the roundtrip but never produced length_test.md, and its test
set did not actually contain all 20 amino acids — it warned about that at runtime instead
of failing. Both are fixed here: every one of the 20 is tested individually, and a missing
one is an error, not a warning.
"""

import os
import sys
import argparse

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common.model_utils import (
    load_model_and_tokenizer, generate_sequence, standard_aa_token_ids,
    START_TOKEN, END_TOKEN, MODEL_NAME,
)
from common.validity import STANDARD_AMINO_ACIDS, MIN_LENGTH, MAX_LENGTH, is_valid_sequence

# Multi-residue sequences that between them cover all 20, plus realistic AMP examples.
TEST_SEQUENCES = [
    "ACDEFGHIKL",
    "MNPQRSTVWY",
    "GIGKFLHSAKKFGKAFVGEIMNS",        # magainin-like
    "KWKLFKKIGAVLKVLTTG",             # cecropin-melittin-like
    "AAAAAAAA",                        # boundary: minimum length
    "A" * MAX_LENGTH,                  # boundary: maximum length
    "ACDEFGHIKLMNPQRSTVWY",           # all 20 in one sequence
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--tag", default="")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip the short-sequence generation part of the length test.")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    model, tokenizer, device = load_model_and_tokenizer(model_name=args.model_name)

    # ---- part 1: every single amino acid on its own -------------------------
    rows = []
    single_failures = []
    for aa in sorted(STANDARD_AMINO_ACIDS):
        enc = tokenizer.encode(aa)
        dec = tokenizer.decode(enc.ids)
        ok = dec == aa
        if not ok:
            single_failures.append(aa)
        rows.append({
            "test": "single_residue",
            "original": aa,
            "encoded_ids": enc.ids,
            "n_tokens": len(enc.ids),
            "decoded_raw": dec,
            "decoded_stripped": dec,
            "roundtrip_matches": ok,
        })

    # ---- part 2: multi-residue sequences with start/end markers -------------
    for seq in TEST_SEQUENCES:
        prompted = START_TOKEN + seq + END_TOKEN
        enc = tokenizer.encode(prompted)
        dec = tokenizer.decode(enc.ids)
        stripped = dec.replace(START_TOKEN, "").replace(END_TOKEN, "").strip()
        rows.append({
            "test": "with_markers",
            "original": seq,
            "encoded_ids": enc.ids,
            "n_tokens": len(enc.ids),
            "decoded_raw": dec,
            "decoded_stripped": stripped,
            "roundtrip_matches": stripped == seq,
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, "docs", f"tokenizer_test{suffix}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(df[["test", "original", "decoded_stripped", "roundtrip_matches"]].to_string(index=False))
    all_ok = bool(df["roundtrip_matches"].all())
    print(f"\nAll roundtrips matched: {all_ok}")

    # ---- part 3: vocab coverage ---------------------------------------------
    vocab = tokenizer.get_vocab()
    missing_from_vocab = sorted(aa for aa in STANDARD_AMINO_ACIDS if aa not in vocab)
    try:
        aa_ids = standard_aa_token_ids(tokenizer)
        ids_ok = len(aa_ids) == 20
    except ValueError as e:
        aa_ids, ids_ok = [], False
        print(f"\nERROR: {e}")

    # ---- part 4: length behaviour (length_test.md) ---------------------------
    length_rows = []
    if not args.skip_generation:
        print("\nGenerating short sequences to test the min-length guard...")
        for min_len in [MIN_LENGTH, 12, 20]:
            for i in range(5):
                seq = generate_sequence(
                    model, tokenizer, device,
                    max_len=MAX_LENGTH, min_len=min_len, seed=100 + i,
                )
                valid, reason = is_valid_sequence(seq)
                length_rows.append({
                    "requested_min_len": min_len,
                    "seed": 100 + i,
                    "generated_length": len(seq),
                    "meets_min": len(seq) >= min_len,
                    "within_max": len(seq) <= MAX_LENGTH,
                    "valid": valid,
                    "reason": reason,
                    "sequence": seq,
                })
                print(f"  min_len={min_len} seed={100+i} -> len {len(seq)} valid={valid}")

    guard_ok = all(r["meets_min"] and r["within_max"] for r in length_rows) if length_rows else None

    lines = [
        f"# Length and tokenizer behaviour — {args.model_name}",
        "",
        "## Amino-acid coverage",
        "",
        f"- Standard amino acids tested individually: {len(STANDARD_AMINO_ACIDS)}/20",
        f"- Missing from the tokenizer vocab: {missing_from_vocab or 'none'}",
        f"- Single-residue roundtrip failures: {single_failures or 'none'}",
        f"- Resolved standard-AA token ids: {len(aa_ids)} "
        f"({'complete' if ids_ok else 'INCOMPLETE — do not generate until fixed'})",
        "",
        "## Start/end token behaviour",
        "",
        f"- Start marker: `{START_TOKEN}`, end marker: `{END_TOKEN}`",
        f"- All roundtrips (including marker stripping) matched: **{all_ok}**",
        "- Exported sequences are stripped of both markers by "
        "`common/model_utils.generate_sequence`.",
        "",
        "## Length rules",
        "",
        f"- Challenge-valid range: {MIN_LENGTH}–{MAX_LENGTH} residues.",
        "- The sampler masks the end token until `min_len` residues exist, so a shorter",
        "  sequence cannot be emitted, and stops at `max_len` regardless.",
        "",
    ]

    if length_rows:
        lines += [
            "### Short-sequence generation test",
            "",
            "| requested min_len | seed | generated length | meets min | within max | valid |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in length_rows:
            lines.append(
                f"| {r['requested_min_len']} | {r['seed']} | {r['generated_length']} | "
                f"{r['meets_min']} | {r['within_max']} | {r['valid']} |"
            )
        lines += ["", f"**Min-length guard held in every case: {guard_ok}**", ""]
        lines += [
            "Note: because the guard makes short sequences unsamplable, a 100% pass here",
            "reflects the guard working, not the model's natural length distribution. The",
            "unguarded distribution is visible in the sampling grid's mean-length column.",
        ]
    else:
        lines += ["_Generation test skipped (--skip-generation)._"]

    out_md = os.path.join(ROOT, "docs", f"length_test{suffix}.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))

    if length_rows:
        pd.DataFrame(length_rows).to_csv(
            os.path.join(ROOT, "docs", f"length_test{suffix}.csv"), index=False
        )

    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_md}")

    if not all_ok or missing_from_vocab or single_failures or not ids_ok:
        print("\nDO NOT proceed to 03_smoke_test.py until the failures above are resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
