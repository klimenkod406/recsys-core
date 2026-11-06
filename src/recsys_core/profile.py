"""Taste profiles: what a user likes, and how much that has faded.

A profile is a weighted vector over the items a user interacted with. The
weight of an interaction decays with a **half life**, so a like from last week
counts more than a like from last quarter::

    decay = 0.5 ** (days_old / HALF_LIFE_DAYS)

With a 30-day half life: an event today is worth 1.0, one 30 days old is
worth 0.5, one 120 days old is worth 0.06.

Signals are weighted by how deliberate they were. A like is a strong signal;
an early skip is a weak negative one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

#: Days after which an interaction counts half as much.
HALF_LIFE_DAYS = 30.0

#: How much each kind of interaction contributes.
WEIGHT_LIKE = 3.0
WEIGHT_PLAYLIST = 2.0
WEIGHT_FOLLOW_ARTIST = 1.5
WEIGHT_COMPLETED_PLAY = 1.0
WEIGHT_PARTIAL_PLAY = 0.3
WEIGHT_VIDEO_COMPLETED = 1.0
WEIGHT_VIDEO_PARTIAL = 0.3
WEIGHT_EXPLICIT_DISLIKE = -1.5
WEIGHT_EARLY_SKIP = -0.8

#: A play shorter than this counts as an early skip, not a real listen.
EARLY_SKIP_SECONDS = 10

#: How far back the profile looks. Six half-lives is roughly "everything
#: that still matters".
LOOKBACK_DAYS = 180

#: Signal name -> weight. Extend or replace per deployment.
DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "like": WEIGHT_LIKE,
    "playlist": WEIGHT_PLAYLIST,
    "follow_artist": WEIGHT_FOLLOW_ARTIST,
    "completed_play": WEIGHT_COMPLETED_PLAY,
    "partial_play": WEIGHT_PARTIAL_PLAY,
    "video_completed": WEIGHT_VIDEO_COMPLETED,
    "video_partial": WEIGHT_VIDEO_PARTIAL,
    "explicit_dislike": WEIGHT_EXPLICIT_DISLIKE,
    "early_skip": WEIGHT_EARLY_SKIP,
}


@dataclass
class TasteProfile:
    """A weighted view of what a user likes.

    Attributes:
        item_weights: ``item_id -> accumulated weight``.
        signal_counts: ``signal name -> number of times it occurred``.
        updated_at: When the profile was last built.
    """

    item_weights: dict[int, float] = field(default_factory=dict)
    signal_counts: dict[str, int] = field(default_factory=dict)
    updated_at: datetime | None = None

    @property
    def size(self) -> int:
        """Number of items with a non-zero weight."""
        return len(self.item_weights)

    def top_items(self, limit: int = 50) -> list[tuple[int, float]]:
        """The heaviest-weighted items, strongest first."""
        return sorted(self.item_weights.items(), key=lambda kv: -kv[1])[:limit]


def time_decay(
    event_dt: datetime | None, now: datetime | None = None
) -> float:
    """How much an interaction from ``event_dt`` still counts.

    Returns 1.0 for an event with no timestamp (treated as "now") and for a
    future event, so that clock skew never produces a negative weight.

    Args:
        event_dt: When the interaction happened.
        now: Reference time; defaults to the current UTC time.

    Returns:
        A multiplier in ``(0, 1]``.
    """
    if event_dt is None:
        return 1.0
    now = now or datetime.now(timezone.utc)
    days_old = max(0.0, (now - event_dt).total_seconds() / 86400.0)
    return 0.5 ** (days_old / HALF_LIFE_DAYS)


def signal_weight(
    signal: str,
    weights: dict[str, float] | None = None,
) -> float:
    """The weight of one interaction signal.

    Unknown signals weigh 0 rather than raising, so a new signal type can be
    logged before it is modelled.
    """
    table = weights if weights is not None else DEFAULT_SIGNAL_WEIGHTS
    return float(table.get(signal, 0.0))


def accumulate(
    profile: TasteProfile,
    item_id: int,
    signal: str,
    event_dt: datetime | None = None,
    now: datetime | None = None,
    weights: dict[str, float] | None = None,
) -> TasteProfile:
    """Fold one interaction into a profile.

    The contribution is ``signal_weight(signal) * time_decay(event_dt)``.

    Args:
        profile: The profile to update in place.
        item_id: The item the interaction was about.
        signal: The kind of interaction.
        event_dt: When it happened.
        now: Reference time for the decay.
        weights: Override for the signal weight table.

    Returns:
        The same profile, for chaining.
    """
    contribution = signal_weight(signal, weights) * time_decay(event_dt, now)
    if contribution:
        item_id = int(item_id)
        profile.item_weights[item_id] = (
            profile.item_weights.get(item_id, 0.0) + contribution
        )
    profile.signal_counts[signal] = profile.signal_counts.get(signal, 0) + 1
    profile.updated_at = now or datetime.now(timezone.utc)
    return profile


def build_profile(
    events: list[tuple[int, str, datetime | None]],
    now: datetime | None = None,
    weights: dict[str, float] | None = None,
) -> TasteProfile:
    """Build a profile from a list of ``(item_id, signal, timestamp)`` events.

    Args:
        events: The interactions to fold in.
        now: Reference time for the decay.
        weights: Override for the signal weight table.

    Returns:
        The built profile.
    """
    profile = TasteProfile()
    for item_id, signal, event_dt in events:
        accumulate(profile, item_id, signal, event_dt, now, weights)
    return profile


def profile_similarity(
    a: dict[int, float], b: dict[int, float]
) -> float:
    """Cosine similarity between two weighted item profiles.

    Returns 0.0 when either profile is empty, rather than raising.
    """
    if not a or not b:
        return 0.0

    shared = set(a) & set(b)
    if not shared:
        return 0.0

    num = sum(a[i] * b[i] for i in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return num / (norm_a * norm_b)
