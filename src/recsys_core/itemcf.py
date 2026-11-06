"""Item-item collaborative filtering on co-occurrence.

The algorithm, in full:

1. Take each user's *positive* signals - the items they interacted with
   deliberately (likes, adding to a playlist). Implicit signals like plays
   are too noisy for this variant.
2. Co-occurrence: ``count(i, j)`` is the number of users who have **both**
   ``i`` and ``j``.
3. Similarity as a normalised cosine over the binary user-item matrix::

       sim(i, j) = co(i, j) / sqrt(c(i) * c(j))

   where ``c(i)`` is the number of users who have ``i``.
4. Keep the top-N neighbours of each item.
5. Recommend for a user by summing the similarity of their items::

       score(j) = sum over i in seen of sim(i, j),  for j not in seen

   Sparse aggregation: only the neighbours of the items the user has are
   considered.

Everything here is a **pure function** over plain dictionaries, so the whole
algorithm is testable without a database, a server or a fixture.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

DEFAULT_TOP_NEIGHBORS = 50

#: Ignore item pairs that fewer than this many users have in common.
#: A single shared item is usually coincidence, not preference.
MIN_COOCCURRENCE = 2


@dataclass
class ItemCFIndex:
    """A built index: for each item, its top-N neighbours.

    Attributes:
        neighbors: ``item_id -> [(neighbour_id, similarity), ...]``, sorted
            by similarity descending.
        item_freq: ``item_id -> number of users holding that item``.
        n_users: how many users the index was built from.
        built_at: Unix timestamp of the build.
    """

    neighbors: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    item_freq: dict[int, int] = field(default_factory=dict)
    n_users: int = 0
    built_at: float = 0.0

    @property
    def size(self) -> int:
        """Number of items that have at least one neighbour."""
        return len(self.neighbors)


def build_index_from_user_items(
    user_items: dict[int, set[int]],
    top_neighbors: int = DEFAULT_TOP_NEIGHBORS,
    min_cooccurrence: int = MIN_COOCCURRENCE,
) -> ItemCFIndex:
    """Build the neighbour index from a ``user_id -> {item_ids}`` mapping.

    Pure function: everything it needs is the mapping.

    Args:
        user_items: The positive interactions of every user.
        top_neighbors: How many neighbours to keep per item.
        min_cooccurrence: Minimum shared users for a pair to count.

    Returns:
        The built index.
    """
    item_freq: dict[int, int] = defaultdict(int)
    cooc: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for items in user_items.values():
        items = list(items)
        for it in items:
            item_freq[int(it)] += 1
        # All unordered pairs.
        for i in range(len(items)):
            a = int(items[i])
            row_a = cooc[a]
            for j in range(i + 1, len(items)):
                b = int(items[j])
                row_a[b] += 1
                cooc[b][a] += 1

    neighbors: dict[int, list[tuple[int, float]]] = {}
    for a, row in cooc.items():
        c_a = item_freq[a]
        if c_a == 0:
            continue
        scored: list[tuple[int, float]] = []
        for b, c_ab in row.items():
            if c_ab < min_cooccurrence:
                continue
            c_b = item_freq.get(b, 0)
            if c_b == 0:
                continue
            sim = c_ab / math.sqrt(c_a * c_b)
            scored.append((b, sim))
        if not scored:
            continue
        scored.sort(key=lambda kv: -kv[1])
        neighbors[a] = scored[:top_neighbors]

    return ItemCFIndex(
        neighbors=neighbors,
        item_freq=dict(item_freq),
        n_users=len(user_items),
        built_at=time.time(),
    )


def recommend_from_seen(
    seen_items: Iterable[int],
    index: ItemCFIndex,
    limit: int = 100,
    exclude: Iterable[int] | None = None,
) -> list[tuple[int, float]]:
    """Score candidates for a user from the items they already have.

    Args:
        seen_items: The user's known items.
        index: A built index.
        limit: Maximum number of candidates to return.
        exclude: Extra items to leave out - dislikes, or anything already
            consumed today. The items in ``seen_items`` are always excluded.

    Returns:
        ``[(item_id, score), ...]``, highest score first.
    """
    scores: dict[int, float] = defaultdict(float)
    seen = {int(i) for i in seen_items}
    excl = {int(i) for i in (exclude or ())} | seen
    for i in seen:
        row = index.neighbors.get(i)
        if not row:
            continue
        for j, sim in row:
            if j in excl:
                continue
            scores[j] += sim
    out = sorted(scores.items(), key=lambda kv: -kv[1])
    return out[:limit]


def similar_items(
    item_id: int, index: ItemCFIndex, limit: int = 10
) -> list[tuple[int, float]]:
    """The top-N items most similar to ``item_id``.

    Used for "more like this" rows.
    """
    row = index.neighbors.get(int(item_id), [])
    return row[:limit]
