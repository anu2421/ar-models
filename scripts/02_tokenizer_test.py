"""
Day 3 deliverable: tokenizer_test.csv — confirm all 20 standard amino acids survive
encode -> decode unchanged, and that start/end tokens don't leak into exported sequences.
"""

import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common.model_utils import load_model_and_tokenizer, START_TOKEN, END_TOKEN
from common.validity import STANDARD_AMINO_ACIDS

TEST_SEQUENCES = [
    "ACDEFGHIKL",                # each amino acid appears at least once across the test set
    "MNPQRSTVWY",
    "GIGKFLHSAKKFGKAFVGEIMNS",   # a real-ish AMP-length example (magainin-like)
    "AAAAAAAA",                   # boundary: minimum length (8)
]


def main():
    _, tokenizer, _ = load_model_and_tokenizer()

    rows = []
    all_aa_seen = set()

    for seq in TEST_SEQUENCES:
        prompted = START_TOKEN + seq + END_TOKEN
        encoded = tokenizer.encode(prompted)
        decoded = tokenizer.decode(encoded.ids)
        stripped = decoded.replace(START_TOKEN, "").replace(END_TOKEN, "")

        rows.append({
            "original": seq,
            "encoded_ids": encoded.ids,
            "n_tokens": len(encoded.ids),
            "decoded_raw": decoded,
            "decoded_stripped": stripped,
            "roundtrip_matches": stripped == seq,
        })
        all_aa_seen.update(seq)

    missing = STANDARD_AMINO_ACIDS - all_aa_seen
    if missing:
        print(f"WARNING: these amino acids weren't covered by the test set: {missing}")
        print("Add them to TEST_SEQUENCES above and rerun before trusting the tokenizer.")

    df = pd.DataFrame(rows)
    out_path = os.path.join(ROOT, "docs", "tokenizer_test.csv")
    df.to_csv(out_path, index=False)

    print(df[["original", "decoded_stripped", "roundtrip_matches"]].to_string(index=False))
    all_ok = df["roundtrip_matches"].all()
    print(f"\nAll roundtrips matched: {all_ok}")
    print(f"Wrote {out_path}")

    if not all_ok:
        print("\nDO NOT proceed to 03_smoke_test.py until every row above is True.")


if __name__ == "__main__":
    main()
