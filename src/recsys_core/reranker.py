"""Re-ranking for a feed: MMR diversity plus hard filters.

The candidate list produced by a retrieval stage is usually ordered by
relevance, which makes it repetitive - ten near-identical items in a row.
Maximal Marginal Relevance trades a little relevance for variety::

    score(c) = lambda * relevance(c) - (1 - lambda) * max_{s in selected} sim(c, s)

``lambda = 1`` is pure relevance, ``lambda = 0`` is pure diversity. The
default of ``0.7`` keeps relevance in charge while breaking up runs.

The filters here are the hard constraints that must hold before diversity is
even considered: nothing the user already consumed, nothing they disliked,
nothing by an artist they blocked.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    The epsilon in the denominator keeps a zero vector from producing a
    division by zero; it yields a similarity of 0, which is the right
    answer for "no direction at all".
    """
    na = float(np.linalg.norm(a)) + 1e-9
    nb = float(np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def mmr_select(
    candidates: list[tuple[int, float]],
    embeddings: dict[int, np.ndarray],
    limit: int,
    lambda_diversity: float = 0.7,
    max_per_artist: int | None = 2,
    artist_of: dict[int, int] | None = None,
) -> list[tuple[int, float]]:
    """Greedy MMR selection over a candidate list.

    Args:
        candidates: ``[(item_id, relevance), ...]``.
        embeddings: ``item_id -> vector``, used to measure similarity. Items
            without an embedding get no diversity penalty.
        limit: How many items to select.
        lambda_diversity: Weight of relevance against diversity.
        max_per_artist: Cap on how many items one artist may contribute.
        artist_of: ``item_id -> artist_id``.

    Returns:
        The selected ``[(item_id, relevance), ...]``.

    Complexity: ``O(K * N)`` with ``N = len(candidates)`` and ``K = limit``.
    """
    if not candidates:
        return []

    artist_of = artist_of or {}
    by_id = {tid: rel for tid, rel in candidates}

    selected: list[tuple[int, float]] = []
    selected_embs: list[np.ndarray] = []
    artist_counts: dict[int, int] = {}
    remaining = list(candidates)

    while remaining and len(selected) < limit:
        best_idx = -1
        best_score = -1e18
        for i, (tid, rel) in enumerate(remaining):
            # Skip items whose artist is already over the cap.
            if max_per_artist is not None:
                aid = artist_of.get(tid)
                if aid is not None and artist_counts.get(aid, 0) >= max_per_artist:
                    continue
            emb = embeddings.get(tid)
            if not selected_embs or emb is None:
                penalty = 0.0
            else:
                penalty = max(cosine_sim(emb, s) for s in selected_embs)
            score = lambda_diversity * rel - (1.0 - lambda_diversity) * penalty
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        tid, rel = remaining.pop(best_idx)
        selected.append((tid, rel))
        emb = embeddings.get(tid)
        if emb is not None:
            selected_embs.append(emb)
        if max_per_artist is not None:
            aid = artist_of.get(tid)
            if aid is not None:
                artist_counts[aid] = artist_counts.get(aid, 0) + 1

    return selected


def filter_listened_today(
    candidates: Iterable[tuple[int, float]],
    listened_today_ids: Iterable[int],
) -> list[tuple[int, float]]:
    """Drop items already consumed today (anti-repeat)."""
    excluded = {int(i) for i in listened_today_ids}
    return [(tid, s) for tid, s in candidates if tid not in excluded]


def filter_explicit_dislikes(
    candidates: Iterable[tuple[int, float]],
    disliked_ids: Iterable[int],
) -> list[tuple[int, float]]:
    """Drop items the user explicitly disliked."""
    excluded = {int(i) for i in disliked_ids}
    return [(tid, s) for tid, s in candidates if tid not in excluded]


def filter_seen_items(
    candidates: Iterable[tuple[int, float]],
    seen_ids: Iterable[int],
) -> list[tuple[int, float]]:
    """Drop items the user has already seen."""
    excluded = {int(i) for i in seen_ids}
    return [(tid, s) for tid, s in candidates if tid not in excluded]


def filter_blocked_artists(
    candidates: Iterable[tuple[int, float]],
    artist_of: dict[int, int],
    blocked_artist_ids: Iterable[int],
) -> list[tuple[int, float]]:
    """Drop items by artists the user blocked."""
    blocked = {int(i) for i in blocked_artist_ids}
    out = []
    for tid, s in candidates:
        aid = artist_of.get(tid)
        if aid is not None and int(aid) in blocked:
            continue
        out.append((tid, s))
    return out
