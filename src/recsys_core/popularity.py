"""Popularity and trending scores.

The trending score is the Hacker News ranking formula::

    (plays - 1) / (age_hours + 2) ** gravity

Two details matter:

* ``plays - 1`` keeps an item with zero plays at zero instead of producing a
  negative score.
* ``+ 2`` in the denominator keeps a brand-new item from dominating on a
  single play, and it is what makes the score comparable across ages.

``gravity = 1.8`` is the Hacker News default; higher values favour newer
items more aggressively.
"""

from __future__ import annotations

GRAVITY = 1.8
DEFAULT_WINDOW_DAYS = 30


def hn_score(plays: int, hours_age: float, gravity: float = GRAVITY) -> float:
    """Rank an item by popularity, discounted by age.

    Args:
        plays: How many times the item was played.
        hours_age: How many hours since the item appeared.
        gravity: How sharply the score decays with age.

    Returns:
        The trending score. Zero for an item with no plays.
    """
    if plays <= 0:
        return 0.0
    age_factor = (max(0.0, hours_age) + 2.0) ** gravity
    return (plays - 1) / age_factor


def trending_score(
    plays: int, hours_age: float, gravity: float = GRAVITY
) -> float:
    """Alias of :func:`hn_score`, for readability at call sites."""
    return hn_score(plays, hours_age, gravity)
