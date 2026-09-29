"""
The novelty / overlap deliverable — the challenge's completion criterion
"Training matches, duplicates and family collapse were measured."

Takes any candidate CSV produced by 03_smoke_test.py, 06_generate_from_checkpoint.py or
08_sampling_grid.py, and measures it against the training split of the AR view.

Usage:
    python scripts/07_overlap_check.py \
        --candidates outputs/finetuned_candidates.csv \
        --views-dir /content/AMP/data/processed/views \
        --tag finetuned

Writes:
    outputs/overlap_<tag>.csv        — per-candidate: exact/near match, nearest neighbour
    outputs/overlap_<tag>.json       — aggregate rates + family-collapse stats
    docs/novelty_report_<tag>.md     — the human-readable report to hand to the team

Run this on EVERY pool before you export candidates. A 95%-valid pool that is 60% near
copies of the training set is worse than a 70%-valid pool that is novel.
"""

import os
import sys
import json
import argparse

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common.data_view import load_ar_view, resolve_data_version
from common.novelty import measure_overlap, measure_internal_redundancy, measure_group_coverage
from common.validity import summarize_validity

TRAIN_SPLITS = {"core_train_only", "train"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True,
                        help="CSV with a 'sequence' column")
    parser.add_argument("--views-dir", required=True)
    parser.add_argument("--view-path", default=None)
    parser.add_argument("--tag", default="pool",
                        help="Label for the output filenames, e.g. 'baseline' or 'finetuned'")
    parser.add_argument("--max-edits", type=int, default=1,
                        help="Edit distance at or below which a candidate counts as a near "
                             "copy of a training sequence. 1 = single point mutation.")
    parser.add_argument("--identity-threshold", type=float, default=0.9,
                        help="Approximate identity at or above which a candidate is flagged.")
    parser.add_argument("--redundancy-edits", type=int, default=2,
                        help="Edit distance for clustering the candidate pool against itself.")
    parser.add_argument("--all-splits", action="store_true",
                        help="Compare against the ENTIRE view, not just the training splits. "
                             "Use this for a final sanity check — a candidate matching a "
                             "held-out test sequence is also not a new design.")
    args = parser.parse_args()

    # ---- load candidates -----------------------------------------------------
    cand_df = pd.read_csv(args.candidates)
    if "sequence" not in cand_df.columns:
        raise SystemExit(f"{args.candidates} has no 'sequence' column. Columns: {list(cand_df.columns)}")
    candidates = [str(s).strip().upper() for s in cand_df["sequence"].dropna()]
    if not candidates:
        raise SystemExit("No sequences found in the candidate file.")

    # ---- load training corpus ------------------------------------------------
    view = load_ar_view(args.views_dir, path=args.view_path)
    data_version = resolve_data_version(view)

    if args.all_splits:
        ref_df = view
        ref_label = "all splits"
    else:
        ref_df = view[view["split"].isin(TRAIN_SPLITS)]
        ref_label = f"splits {sorted(TRAIN_SPLITS)}"
    training = [str(s).strip().upper() for s in ref_df["sequence"].dropna()]
    if not training:
        raise SystemExit(f"No reference sequences found for {ref_label}.")

    print(f"Comparing {len(candidates)} candidates against {len(training)} sequences ({ref_label}).")

    # ---- measure -------------------------------------------------------------
    overlap = measure_overlap(
        candidates, training,
        max_edits=args.max_edits,
        identity_threshold=args.identity_threshold,
    )
    redundancy = measure_internal_redundancy(candidates, max_edits=args.redundancy_edits)
    validity = summarize_validity(candidates)

    group_cov = None
    if "similarity_group_id" in ref_df.columns:
        groups = ref_df["similarity_group_id"].tolist()
        # keep training/groups aligned after the dropna above
        aligned = ref_df.dropna(subset=["sequence"])
        group_cov = measure_group_coverage(
            candidates,
            [str(s).strip().upper() for s in aligned["sequence"]],
            aligned["similarity_group_id"].tolist(),
            max_edits=max(args.max_edits, 2),
        )
    else:
        print("NOTE: view has no similarity_group_id column — skipping family-coverage stats.")

    # ---- write ---------------------------------------------------------------
    per_cand = pd.DataFrame(overlap["per_candidate"])
    csv_path = os.path.join(ROOT, "outputs", f"overlap_{args.tag}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    per_cand.to_csv(csv_path, index=False)

    report = {
        "candidate_file": os.path.relpath(args.candidates, ROOT) if os.path.isabs(args.candidates) else args.candidates,
        "reference": ref_label,
        "data_version": data_version,
        "validity": validity,
        "training_overlap": overlap["summary"],
        "internal_redundancy": redundancy,
        "group_coverage": group_cov,
    }
    json_path = os.path.join(ROOT, "outputs", f"overlap_{args.tag}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # ---- markdown report -----------------------------------------------------
    s = overlap["summary"]
    lines = [
        f"# Novelty report — {args.tag}",
        "",
        f"- Candidate file: `{args.candidates}`",
        f"- Compared against: {ref_label} of the AR view (`{len(training)}` sequences)",
        f"- data_version: `{data_version}`",
        f"- Near-match threshold: edit distance <= {args.max_edits}; "
        f"identity flag at >= {args.identity_threshold}",
        "",
        "## Headline numbers",
        "",
        "| Metric | Count | Rate |",
        "| --- | --- | --- |",
        f"| Candidates | {s['n_candidates']} | — |",
        f"| Valid (challenge rules) | {validity['n_valid']} | {validity['validity_rate']:.1%} |",
        f"| Exact training matches | {s['n_exact_matches']} | {s['exact_match_rate']:.1%} |",
        f"| Near training matches (<= {args.max_edits} edit) | {s['n_near_matches']} | {s['near_match_rate']:.1%} |",
        f"| High identity (>= {args.identity_threshold}) | {s['n_high_identity']} | {s['high_identity_rate']:.1%} |",
        f"| **Novel** | **{s['n_novel']}** | **{s['novelty_rate']:.1%}** |",
        "",
        "## Family collapse within the pool",
        "",
        f"- Exact duplicates: {redundancy['exact_duplicate_count']}",
        f"- Clusters at edit distance <= {args.redundancy_edits}: "
        f"{redundancy['n_clusters']} for {redundancy['n_candidates']} candidates",
        f"- Effective diversity (clusters / candidates): **{redundancy['effective_diversity']}** "
        f"(1.0 = every candidate its own family)",
        f"- Largest single family: {redundancy['largest_cluster_size']} candidates",
        f"- Singleton rate: {redundancy['singleton_rate']:.1%}",
    ]
    if redundancy["top_clusters"] and redundancy["largest_cluster_size"] > 1:
        lines += ["", "Largest families (representative sequence):", ""]
        for c in redundancy["top_clusters"]:
            if c["size"] > 1:
                lines.append(f"- `{c['representative']}` — {c['size']} members")

    if group_cov:
        lines += [
            "",
            "## Coverage of training similarity groups",
            "",
            f"- Training groups hit: {group_cov['n_training_groups_hit']} / "
            f"{group_cov['n_training_groups_total']} ({group_cov['group_coverage_rate']:.1%})",
            f"- Candidates not attributable to any training group: "
            f"{group_cov['n_unattributed_candidates']} ({group_cov['unattributed_rate']:.1%}) "
            f"— a high number here is a GOOD novelty signal",
        ]

    lines += [
        "",
        "## How to read this",
        "",
        "A fluent copy is not a new design. Exact and near matches should both be low; if the",
        "near-match rate is high while validity looks great, the model has memorised the",
        "training corpus rather than learned the design space. Effective diversity well below",
        "1.0 means the pool is a handful of families with point mutations — fix that by raising",
        "sampling temperature or reducing fine-tuning epochs, not by generating more sequences.",
        "",
        "Caveat: identity here is `1 - normalised edit distance`, not a Smith-Waterman",
        "alignment identity. Do not quote these numbers as alignment identities in the",
        "final write-up.",
    ]

    md_path = os.path.join(ROOT, "docs", f"novelty_report_{args.tag}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    # ---- console -------------------------------------------------------------
    print("\n--- Training overlap ---")
    print(json.dumps(s, indent=2))
    print("\n--- Internal redundancy ---")
    print(json.dumps({k: v for k, v in redundancy.items() if k != "top_clusters"}, indent=2))
    if group_cov:
        print("\n--- Group coverage ---")
        print(json.dumps({k: v for k, v in group_cov.items() if k != "most_hit_groups"}, indent=2))

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    if s["exact_match_rate"] > 0.01:
        print("\nWARNING: more than 1% of candidates are verbatim training sequences.")
    if s["near_match_rate"] > 0.20:
        print("\nWARNING: over 20% of candidates are within "
              f"{args.max_edits} edit of a training sequence — likely memorisation.")
    if redundancy["effective_diversity"] < 0.5:
        print("\nWARNING: effective diversity below 0.5 — the pool has collapsed into families.")


if __name__ == "__main__":
    main()
