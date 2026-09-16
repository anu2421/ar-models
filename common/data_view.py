"""
Locate and load the autoregressive-specific data view from the Data Engineering repo
(their README: `data/processed/views/` holds explicit per-role views — autoregressive,
VAE/latent, diffusion, evolution-seed, evaluator).

We don't know the AR view's exact filename in advance, so this searches for it instead
of hard-coding a guess that might be wrong.
"""

import glob
import os
import pandas as pd

AR_VIEW_PATTERNS = [
    "*autoregressive*",
    "*_ar_*",
    "ar_view*",
    "*ar*view*",
]

REQUIRED_COLUMNS = ["sequence", "length", "split", "similarity_group_id", "data_version"]


def find_ar_view(views_dir: str) -> str:
    """
    Search the views directory for a file that looks like the AR view.
    Raises FileNotFoundError with the full directory listing if nothing matches, so you
    can see what's actually there and fix AR_VIEW_PATTERNS instead of guessing blind.
    """
    if not os.path.isdir(views_dir):
        raise FileNotFoundError(f"No such directory: {views_dir}")

    candidates = []
    for pattern in AR_VIEW_PATTERNS:
        candidates.extend(glob.glob(os.path.join(views_dir, pattern)))
    candidates = sorted(set(c for c in candidates if os.path.isfile(c)))

    if not candidates:
        listing = "\n".join(sorted(os.listdir(views_dir)))
        raise FileNotFoundError(
            f"Couldn't auto-detect the AR view in {views_dir}.\n"
            f"Files actually present:\n{listing}\n"
            "Add the real filename pattern to AR_VIEW_PATTERNS in common/data_view.py."
        )

    if len(candidates) > 1:
        print(f"WARNING: multiple candidate AR view files found, using the first: {candidates}")

    return candidates[0]


def load_ar_view(views_dir: str) -> pd.DataFrame:
    path = find_ar_view(views_dir)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unrecognized file type: {path}")
    print(f"Loaded AR view from: {path}")
    return df


def check_schema(df: pd.DataFrame) -> dict:
    present = set(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    extra = sorted(present - set(REQUIRED_COLUMNS))
    return {"missing_required": missing, "extra_columns": extra, "n_rows": len(df)}
