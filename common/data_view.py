"""
Locate and load the autoregressive-specific data view from the Data Engineering repo
(their README: `data/processed/views/` holds explicit per-role views — autoregressive,
VAE/latent, diffusion, evolution-seed, evaluator).

We don't know the AR view's exact filename in advance, so this searches for it instead
of hard-coding a guess that might be wrong.

Matching uses token boundaries, NOT bare substrings. The earlier pattern `*ar*view*`
matched "simil-AR-ity_group_VIEW.parquet", and because parquet is preferred over csv it
picked that file over the real `view_autoregressive.csv`. Data Engineering ships a
similarity-group file as a named deliverable, so that collision was likely to fire in
practice. Fix: a filename qualifies only if one of its underscore/dash/dot-delimited
tokens is exactly "ar", or if it contains the full word "autoregressive".
"""

import glob
import os
import re

import pandas as pd

# A filename matches if EITHER regex matches (case-insensitive).
#   1. the whole word "autoregressive" appears anywhere
#   2. "ar" appears as a standalone token, delimited by _ - . or string boundaries
AR_VIEW_REGEXES = [
    re.compile(r"autoregressive", re.IGNORECASE),
    re.compile(r"(?:^|[_\-.])ar(?:$|[_\-.])", re.IGNORECASE),
]

# Anything matching one of these is never the AR view, even if a rule above fires.
# Cheap insurance against another role's file being picked up by accident.
EXCLUDE_REGEXES = [
    re.compile(r"similar", re.IGNORECASE),
    re.compile(r"cluster", re.IGNORECASE),
    re.compile(r"\bvae\b|latent", re.IGNORECASE),
    re.compile(r"diffusion", re.IGNORECASE),
    re.compile(r"evaluator", re.IGNORECASE),
    re.compile(r"evolution", re.IGNORECASE),
]

DATA_EXTENSIONS = (".parquet", ".csv", ".tsv")

REQUIRED_COLUMNS = ["sequence", "length", "split", "similarity_group_id", "data_version"]

# Some Data Engineering builds use different column names for the same concept.
# Map: our expected name -> alternate names we've actually seen in the wild.
# load_ar_view() RENAMES these in the returned DataFrame, so downstream code can always
# use the canonical name. (Previously check_schema only *reported* the alias, which made
# the schema check pass while df["similarity_group_id"] still raised KeyError.)
COLUMN_ALIASES = {
    "similarity_group_id": ["cluster_id", "similarity_cluster_id", "group_id"],
    "sequence": ["seq", "peptide", "peptide_sequence"],
    "length": ["seq_len", "sequence_length"],
    "split": ["split_name", "partition"],
    "data_version": ["version", "dataset_version"],
}


def _looks_like_ar_view(filename: str) -> bool:
    name = os.path.basename(filename)
    if any(rx.search(name) for rx in EXCLUDE_REGEXES):
        return False
    return any(rx.search(name) for rx in AR_VIEW_REGEXES)


def find_ar_view(views_dir: str) -> str:
    """
    Search the views directory for a file that looks like the AR view.
    Raises FileNotFoundError with the full directory listing if nothing matches, so you
    can see what's actually there and fix the matching rules instead of guessing blind.
    Prefers .parquet over .csv when both are present (parquet preserves dtypes).
    """
    if not os.path.isdir(views_dir):
        raise FileNotFoundError(f"No such directory: {views_dir}")

    all_files = [
        p for p in glob.glob(os.path.join(views_dir, "*"))
        if os.path.isfile(p) and p.lower().endswith(DATA_EXTENSIONS)
    ]
    candidates = sorted(p for p in all_files if _looks_like_ar_view(p))

    if not candidates:
        listing = "\n".join(f"  {f}" for f in sorted(os.listdir(views_dir)))
        raise FileNotFoundError(
            f"Couldn't auto-detect the AR view in {views_dir}.\n"
            f"Files actually present:\n{listing}\n\n"
            "Either ask Data Engineering for the exact AR view filename, or add a rule to "
            "AR_VIEW_REGEXES in common/data_view.py. Do not rename their file — that "
            "breaks traceability back to their data_version."
        )

    parquet_candidates = [c for c in candidates if c.endswith(".parquet")]
    chosen = parquet_candidates[0] if parquet_candidates else candidates[0]

    if len(candidates) > 1:
        print(
            "NOTE: multiple candidate AR view files matched:\n"
            + "\n".join(f"  {os.path.basename(c)}" for c in candidates)
            + f"\n  -> using {os.path.basename(chosen)}"
            + ("  (parquet preferred)" if parquet_candidates else "")
            + "\n  If that is the wrong file, pass the right one explicitly."
        )

    return chosen


def _apply_aliases(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename known alternate column names to our canonical names. Returns (df, applied)."""
    present = set(df.columns)
    renames = {}
    for canonical, alternates in COLUMN_ALIASES.items():
        if canonical in present:
            continue
        alt = next((a for a in alternates if a in present), None)
        if alt is not None:
            renames[alt] = canonical
    if renames:
        df = df.rename(columns=renames)
        print(f"Applied column aliases: {renames}")
    return df, {v: k for k, v in renames.items()}


def load_ar_view(views_dir: str, path: str | None = None) -> pd.DataFrame:
    """
    Load the AR view and normalise its column names.

    path: bypass auto-detection and load this exact file. Use this whenever Data
    Engineering tells you the real filename — it is always safer than pattern matching.
    """
    path = path or find_ar_view(views_dir)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unrecognized file type: {path}")

    df, applied = _apply_aliases(df)
    df.attrs["source_path"] = path
    df.attrs["applied_aliases"] = applied
    print(f"Loaded AR view from: {path}  ({len(df)} rows)")
    return df


def check_schema(df: pd.DataFrame) -> dict:
    """
    Verify the canonical columns are present. Run this AFTER load_ar_view, which has
    already applied aliases — so anything reported missing here is genuinely absent and
    needs to go back to Data Engineering.
    """
    present = set(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    extra = sorted(present - set(REQUIRED_COLUMNS))
    return {
        "missing_required": missing,
        "applied_aliases": df.attrs.get("applied_aliases", {}),
        "extra_columns": extra,
        "n_rows": len(df),
        "source_path": df.attrs.get("source_path", "unknown"),
    }


def resolve_data_version(df: pd.DataFrame) -> str:
    """
    Single source of truth for the data_version stamped into every manifest.
    Raises if the view carries more than one version — mixing versions in one training
    run destroys the traceability the challenge's completion criteria require.
    """
    if "data_version" not in df.columns:
        return "unknown"
    versions = sorted(str(v) for v in df["data_version"].dropna().unique())
    if not versions:
        return "unknown"
    if len(versions) > 1:
        raise ValueError(
            f"AR view contains multiple data_version values: {versions}. "
            "Ask Data Engineering for a single frozen version before training."
        )
    return versions[0]
