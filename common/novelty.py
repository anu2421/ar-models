"""
Novelty and overlap measurement — "a fluent copy is not a new design."

This is the module that answers the challenge's completion criterion:
"Training matches, duplicates and family collapse were measured."

Three distinct things get measured, and they are NOT the same:

  1. EXACT overlap      — candidate string appears verbatim in the training corpus.
  2. NEAR overlap       — candidate is within `max_edits` Levenshtein distance of some
                          training sequence, or above an identity threshold. A single
                          point mutation off a known AMP is not a new design.
  3. FAMILY COLLAPSE    — the candidate pool itself is redundant: many candidates are
                          near-duplicates of each other, or they all pile into a handful
                          of training similarity groups.

Pure standard library + numpy, so this runs without torch and can be checked on CPU.

Performance: exact overlap is a set lookup. Near overlap is the expensive part, so it is
bounded two ways — a length-window prefilter (sequences whose lengths differ by more than
max_edits cannot be within max_edits) and a k-mer prefilter, before any edit distance is
computed. That keeps a 1,000 x 27,000 comparison tractable on CPU.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict


# ----------------------------------------------------------------------------------
# edit distance
# ----------------------------------------------------------------------------------

def levenshtein(a: str, b: str, cap: int | None = None) -> int:
    """
    Standard Levenshtein distance, two-row DP. If `cap` is given, returns early with
    cap + 1 as soon as the whole row exceeds cap — we only ever care whether a candidate
    is *within* a small distance, so there is no reason to finish computing a distance of 30.
    """
    if a == b:
        return 0
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1,          # deletion
                cur[j - 1] + 1,       # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        if cap is not None and min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def identity_ratio(a: str, b: str) -> float:
    """
    Fraction of positions that match, as 1 - normalised edit distance. Reported as an
    approximate sequence identity — this is NOT a Smith-Waterman alignment identity, and
    should not be quoted as one in a write-up. For peptides of similar length it tracks
    alignment identity closely enough to flag near-copies.
    """
    if not a and not b:
        return 1.0
    d = levenshtein(a, b)
    return 1.0 - d / max(len(a), len(b))


# ----------------------------------------------------------------------------------
# k-mer index for prefiltering
# ----------------------------------------------------------------------------------

def kmers(seq: str, k: int = 4) -> set[str]:
    if len(seq) < k:
        return {seq}
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


class KmerIndex:
    """Maps each k-mer to the set of training-sequence indices containing it."""

    def __init__(self, sequences: list[str], k: int = 4):
        self.k = k
        self.sequences = sequences
        self.index: dict[str, set[int]] = defaultdict(set)
        self.by_length: dict[int, set[int]] = defaultdict(set)
        for i, s in enumerate(sequences):
            self.by_length[len(s)].add(i)
            for km in kmers(s, k):
                self.index[km].add(i)

    def candidates_within(self, seq: str, max_edits: int, min_shared_kmers: int = 1) -> set[int]:
        """
        Training indices that could plausibly be within max_edits of `seq`.
        Two prefilters, both of which are safe (they can only over-include):
          - length window: |len difference| <= max_edits
          - shared k-mers: a sequence within e edits shares at least some k-mers
        """
        length_ok: set[int] = set()
        for L in range(len(seq) - max_edits, len(seq) + max_edits + 1):
            length_ok |= self.by_length.get(L, set())
        if not length_ok:
            return set()

        counts: Counter[int] = Counter()
        for km in kmers(seq, self.k):
            for i in self.index.get(km, ()):
                if i in length_ok:
                    counts[i] += 1
        return {i for i, c in counts.items() if c >= min_shared_kmers}


# ----------------------------------------------------------------------------------
# the measurements
# ----------------------------------------------------------------------------------

def measure_overlap(
    candidates: list[str],
    training: list[str],
    max_edits: int = 1,
    identity_threshold: float = 0.9,
    k: int = 4,
) -> dict:
    """
    Per-candidate overlap against the training corpus.

    Returns a dict with a 'per_candidate' list (one row per candidate, ready to become a
    CSV) and a 'summary' dict of aggregate rates.

    max_edits           — a candidate within this edit distance counts as a NEAR match.
    identity_threshold  — a candidate at or above this identity counts as HIGH IDENTITY,
                          even if it is more than max_edits away (relevant for long peptides).
    """
    training_set = set(training)
    index = KmerIndex(training, k=k)

    rows = []
    for cand in candidates:
        exact = cand in training_set

        nearest_d = None
        nearest_seq = None
        if exact:
            nearest_d, nearest_seq = 0, cand
        else:
            # search a widened window so identity can be computed even when the
            # candidate is further than max_edits away
            search_edits = max(max_edits, int(round((1 - identity_threshold) * max(1, len(cand)))) + 1)
            for i in index.candidates_within(cand, search_edits):
                d = levenshtein(cand, training[i], cap=search_edits)
                if nearest_d is None or d < nearest_d:
                    nearest_d, nearest_seq = d, training[i]
                    if d == 0:
                        break

        if nearest_d is None:
            near = False
            ident = 0.0
            nearest_d_out = None
        else:
            near = nearest_d <= max_edits
            ident = 1.0 - nearest_d / max(len(cand), len(nearest_seq)) if nearest_seq else 0.0
            nearest_d_out = nearest_d

        rows.append({
            "sequence": cand,
            "length": len(cand),
            "exact_training_match": exact,
            "near_training_match": near,
            "nearest_edit_distance": nearest_d_out,
            "nearest_training_sequence": nearest_seq,
            "approx_identity_to_nearest": round(ident, 4),
            "high_identity_match": ident >= identity_threshold,
            "is_novel": not exact and not near and ident < identity_threshold,
        })

    n = len(candidates)
    def rate(key):
        return round(sum(1 for r in rows if r[key]) / n, 4) if n else 0.0

    summary = {
        "n_candidates": n,
        "n_training": len(training),
        "max_edits": max_edits,
        "identity_threshold": identity_threshold,
        "n_exact_matches": sum(1 for r in rows if r["exact_training_match"]),
        "exact_match_rate": rate("exact_training_match"),
        "n_near_matches": sum(1 for r in rows if r["near_training_match"]),
        "near_match_rate": rate("near_training_match"),
        "n_high_identity": sum(1 for r in rows if r["high_identity_match"]),
        "high_identity_rate": rate("high_identity_match"),
        "n_novel": sum(1 for r in rows if r["is_novel"]),
        "novelty_rate": rate("is_novel"),
    }
    return {"per_candidate": rows, "summary": summary}


def measure_internal_redundancy(candidates: list[str], max_edits: int = 1, k: int = 4) -> dict:
    """
    Family collapse WITHIN the candidate pool: how many candidates are near-duplicates of
    each other. A pool of 1,000 sequences that is really 40 designs with point mutations
    has a diversity problem that a plain duplicate count will not show.

    Uses single-linkage clustering at the max_edits threshold. Reports the largest
    clusters, since "one giant family" is the failure mode worth catching.
    """
    n = len(candidates)
    if n == 0:
        return {"n_candidates": 0, "n_exact_unique": 0, "n_clusters": 0,
                "largest_cluster_size": 0, "singleton_rate": 0.0,
                "top_clusters": [], "effective_diversity": 0.0}

    index = KmerIndex(candidates, k=k)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, cand in enumerate(candidates):
        for j in index.candidates_within(cand, max_edits):
            if j <= i:
                continue
            if levenshtein(cand, candidates[j], cap=max_edits) <= max_edits:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)

    top = sorted(clusters.values(), key=len, reverse=True)[:5]
    top_clusters = [
        {"size": len(c), "representative": candidates[c[0]],
         "members_preview": [candidates[i] for i in c[:4]]}
        for c in top
    ]

    return {
        "n_candidates": n,
        "n_exact_unique": len(set(candidates)),
        "exact_duplicate_count": n - len(set(candidates)),
        "n_clusters": len(clusters),
        "largest_cluster_size": sizes[0] if sizes else 0,
        "singleton_rate": round(sum(1 for s in sizes if s == 1) / n, 4),
        # clusters per candidate: 1.0 means every candidate is its own family, low means collapse
        "effective_diversity": round(len(clusters) / n, 4),
        "cluster_size_histogram": dict(Counter(sizes)),
        "top_clusters": top_clusters,
        "max_edits": max_edits,
    }


def measure_group_coverage(
    candidates: list[str],
    training: list[str],
    training_groups: list,
    max_edits: int = 2,
    k: int = 4,
) -> dict:
    """
    Do the candidates spread across the training similarity groups, or pile into a few?
    Needs the AR view's similarity_group_id column, which is exactly why we asked Data
    Engineering for it.

    Each candidate is attributed to the similarity group of its nearest training sequence
    (within max_edits); candidates with no near match are counted as 'unattributed', which
    is a GOOD sign for novelty.
    """
    if len(training) != len(training_groups):
        raise ValueError("training and training_groups must be the same length")

    index = KmerIndex(training, k=k)
    hit_groups: Counter = Counter()
    unattributed = 0

    for cand in candidates:
        best_d, best_group = None, None
        for i in index.candidates_within(cand, max_edits):
            d = levenshtein(cand, training[i], cap=max_edits)
            if d <= max_edits and (best_d is None or d < best_d):
                best_d, best_group = d, training_groups[i]
                if d == 0:
                    break
        if best_group is None:
            unattributed += 1
        else:
            hit_groups[best_group] += 1

    total_groups = len(set(training_groups))
    attributed = sum(hit_groups.values())
    return {
        "n_candidates": len(candidates),
        "n_training_groups_total": total_groups,
        "n_training_groups_hit": len(hit_groups),
        "group_coverage_rate": round(len(hit_groups) / total_groups, 4) if total_groups else 0.0,
        "n_unattributed_candidates": unattributed,
        "unattributed_rate": round(unattributed / len(candidates), 4) if candidates else 0.0,
        "n_attributed_candidates": attributed,
        "most_hit_groups": hit_groups.most_common(10),
        "max_edits": max_edits,
    }
