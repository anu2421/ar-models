"""
Validity rules shared by every domain in the AMP Challenge (see the Data Engineering
guide, Step 4: "challenge-ready sequences use only the 20 standard amino acids, are 8 to
50 residues long, are linear, have free termini and contain no chemical modifications").

Keep this file identical to whatever Data Engineering ships once their view exists —
don't invent your own separate rule set.
"""

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

MIN_LENGTH = 8
MAX_LENGTH = 50


def is_valid_sequence(sequence: str, min_len: int = MIN_LENGTH, max_len: int = MAX_LENGTH):
    """
    Returns (is_valid: bool, reason: str).
    reason is one of: "valid", "invalid_length", "non_standard_amino_acid", "empty_sequence".
    """
    if not sequence:
        return False, "empty_sequence"
    if not (min_len <= len(sequence) <= max_len):
        return False, "invalid_length"
    if not set(sequence).issubset(STANDARD_AMINO_ACIDS):
        return False, "non_standard_amino_acid"
    return True, "valid"


def summarize_validity(sequences: list[str]) -> dict:
    """Quick aggregate stats for a smoke test — validity rate, reason breakdown, duplicates."""
    results = [is_valid_sequence(s) for s in sequences]
    n = len(sequences)
    n_valid = sum(1 for ok, _ in results if ok)
    reason_counts: dict[str, int] = {}
    for _, reason in results:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    n_duplicates = n - len(set(sequences))
    return {
        "n_generated": n,
        "n_valid": n_valid,
        "validity_rate": round(n_valid / n, 4) if n else 0.0,
        "reason_counts": reason_counts,
        "n_duplicates": n_duplicates,
        "n_unique": len(set(sequences)),
    }
