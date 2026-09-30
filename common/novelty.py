"""
Novelty and overlap measurement.

Three separate things, which are not the same:
  exact overlap    candidate appears verbatim in the training corpus
  near overlap     candidate is within max_edits of a training sequence
  family collapse  the candidate pool is internally redundant

Near overlap is the expensive one, so it is bounded by a length window and a k-mer
prefilter before any edit distance is computed. Both prefilters can only over-include,
never miss a true match. Pure stdlib, so this runs without torch.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def levenshtein(a: str, b: str, cap: int | None = None) -> int:
    """Edit distance, two-row DP. With cap, returns cap+1 as soon as the distance exceeds it."""
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
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if cap is not None and min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def identity_ratio(a: str, b: str) -> float:
    """1 - normalised edit distance. Not an alignment identity; do not report it as one."""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def kmers(seq: str, k: int = 4) -> set[str]:
    if len(seq) < k:
        return {seq}
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


class KmerIndex:
    """Maps each k-mer to the training indices containing it."""

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
        Training indices that could be within max_edits. Over-includes; never misses.

        The k-mer filter is only safe when a sequence has more k-mers than the edits can
        destroy. A sequence of length L has L-k+1 k-mers and each edit can break up to k of
        them, so the filter is dropped (length window only) when L-k+1 <= k*max_edits.
        Without this, an 8-residue peptide at max_edits=2 has 5 k-mers against 8 that can
        break, and true near matches were silently missed: 32% of them at L=8, 15% at L=9,
        7% at L=10, 0% at L>=12.
        """
        length_ok: set[int] = set()
        for L in range(len(seq) - max_edits, len(seq) + max_edits + 1):
            length_ok |= self.by_length.get(L, set())
        if not length_ok:
            return set()

        n_kmers = max(0, len(seq) - self.k + 1)
        if n_kmers - self.k * max_edits < min_shared_kmers:
            return length_ok

        counts: Counter[int] = Counter()
        for km in kmers(seq, self.k):
            for i in self.index.get(km, ()):
                if i in length_ok:
                    counts[i] += 1
        return {i for i, c in counts.items() if c >= min_shared_kmers}


def measure_overlap(
    candidates: list[str],
    training: list[str],
    max_edits: int = 1,
    identity_threshold: float = 0.9,
    k: int = 4,
) -> dict:
    """
    Per-candidate overlap against the training corpus.
    Returns {"per_candidate": [...], "summary": {...}}.
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
            # widen the window so identity is computable past max_edits
            search_edits = max(
                max_edits,
                int(round((1 - identity_threshold) * max(1, len(cand)))) + 1,
            )
            for i in index.candidates_within(cand, search_edits):
                d = levenshtein(cand, training[i], cap=search_edits)
                if nearest_d is None or d < nearest_d:
                    nearest_d, nearest_seq = d, training[i]
                    if d == 0:
                        break

        if nearest_d is None:
            near, ident = False, 0.0
        else:
            near = nearest_d <= max_edits
            ident = 1.0 - nearest_d / max(len(cand), len(nearest_seq)) if nearest_seq else 0.0

        rows.append({
            "sequence": cand,
            "length": len(cand),
            "exact_training_match": exact,
            "near_training_match": near,
            "nearest_edit_distance": nearest_d,
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
    Family collapse within the pool: single-linkage clustering at max_edits.
    A plain duplicate count misses 1000 sequences that are really 40 designs with mutations.
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
            if j > i and levenshtein(cand, candidates[j], cap=max_edits) <= max_edits:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)

    top_clusters = [
        {"size": len(c), "representative": candidates[c[0]],
         "members_preview": [candidates[i] for i in c[:4]]}
        for c in sorted(clusters.values(), key=len, reverse=True)[:5]
    ]

    return {
        "n_candidates": n,
        "n_exact_unique": len(set(candidates)),
        "exact_duplicate_count": n - len(set(candidates)),
        "n_clusters": len(clusters),
        "largest_cluster_size": sizes[0] if sizes else 0,
        "singleton_rate": round(sum(1 for s in sizes if s == 1) / n, 4),
        # 1.0 = every candidate its own family; low = collapse
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
    Do candidates spread across training similarity groups or pile into a few?
    Each candidate is attributed to its nearest training sequence's group; no near match
    means unattributed, which is a good novelty signal.
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
    return {
        "n_candidates": len(candidates),
        "n_training_groups_total": total_groups,
        "n_training_groups_hit": len(hit_groups),
        "group_coverage_rate": round(len(hit_groups) / total_groups, 4) if total_groups else 0.0,
        "n_unattributed_candidates": unattributed,
        "unattributed_rate": round(unattributed / len(candidates), 4) if candidates else 0.0,
        "n_attributed_candidates": sum(hit_groups.values()),
        "most_hit_groups": hit_groups.most_common(10),
        "max_edits": max_edits,
    }