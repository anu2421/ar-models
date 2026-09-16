"""
Day 5-6 sanity check: load the real AR view from the Data Engineering repo and confirm
it has what was requested, before writing any training code against it.

Usage:
    python scripts/04_inspect_ar_view.py --views-dir /path/to/amp-data-repo/data/processed/views
"""

import os
import sys
import argparse
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common.data_view import load_ar_view, check_schema
from common.validity import summarize_validity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--views-dir",
        required=True,
        help="Path to the Data Engineering repo's data/processed/views/ folder",
    )
    args = parser.parse_args()

    df = load_ar_view(args.views_dir)
    schema = check_schema(df)

    print("\n--- Schema check ---")
    print(json.dumps(schema, indent=2))
    if schema["missing_required"]:
        print(
            f"\nMISSING required columns: {schema['missing_required']} "
            "— go back to Data Engineering before building training code on this file."
        )

    print("\n--- Basic stats ---")
    print(f"Rows: {len(df)}")
    if "split" in df.columns:
        print(df["split"].value_counts().to_string())
    if "data_version" in df.columns:
        print(f"data_version value(s) present: {df['data_version'].unique().tolist()}")

    report_lines = [
        "# AR view inspection",
        "",
        f"Source directory: {args.views_dir}",
        f"Rows: {len(df)}",
        f"Columns: {list(df.columns)}",
        "",
        "## Schema check",
        f"```json\n{json.dumps(schema, indent=2)}\n```",
    ]

    if "sequence" in df.columns:
        validity = summarize_validity(df["sequence"].dropna().astype(str).tolist())
        print("\n--- Sequence validity (challenge rules) ---")
        print(json.dumps(validity, indent=2))
        report_lines += [
            "",
            "## Sequence validity",
            f"```json\n{json.dumps(validity, indent=2)}\n```",
        ]

    out_path = os.path.join(ROOT, "docs", "ar_view_inspection.md")
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
