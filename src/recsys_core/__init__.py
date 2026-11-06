"""Recommendation core.

Four pieces of a recommender, each a pure function over plain data:

- :mod:`~recsys_core.itemcf` - item-item collaborative filtering.
- :mod:`~recsys_core.reranker` - MMR diversity re-ranking and hard filters.
- :mod:`~recsys_core.popularity` - Hacker News trending score.
- :mod:`~recsys_core.profile` - time-decayed taste profiles.

Nothing here touches a database, a network or a global cache. Every function
takes its data as an argument and returns a value, which makes the whole
algorithm testable in isolation.
"""

from .itemcf import (
    DEFAULT_TOP_NEIGHBORS,
    MIN_COOCCURRENCE,
    ItemCFIndex,
    build_index_from_user_items,
    recommend_from_seen,
    similar_items,
)
from .popularity import DEFAULT_WINDOW_DAYS, GRAVITY, hn_score, trending_score
from .profile import (
    DEFAULT_SIGNAL_WEIGHTS,
    EARLY_SKIP_SECONDS,
    HALF_LIFE_DAYS,
    LOOKBACK_DAYS,
    TasteProfile,
    accumulate,
    build_profile,
    profile_similarity,
    signal_weight,
    time_decay,
)
from .reranker import (
    cosine_sim,
    filter_blocked_artists,
    filter_explicit_dislikes,
    filter_listened_today,
    filter_seen_items,
    mmr_select,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_TOP_NEIGHBORS",
    "MIN_COOCCURRENCE",
    "ItemCFIndex",
    "build_index_from_user_items",
    "recommend_from_seen",
    "similar_items",
    "DEFAULT_WINDOW_DAYS",
    "GRAVITY",
    "hn_score",
    "trending_score",
    "DEFAULT_SIGNAL_WEIGHTS",
    "EARLY_SKIP_SECONDS",
    "HALF_LIFE_DAYS",
    "LOOKBACK_DAYS",
    "TasteProfile",
    "accumulate",
    "build_profile",
    "profile_similarity",
    "signal_weight",
    "time_decay",
    "cosine_sim",
    "filter_blocked_artists",
    "filter_explicit_dislikes",
    "filter_listened_today",
    "filter_seen_items",
    "mmr_select",
]
