# recsys-core

[![CI](https://github.com/klimenkod406/recsys-core/actions/workflows/ci.yml/badge.svg)](https://github.com/klimenkod406/recsys-core/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

The algorithmic core of a recommender, as **pure functions**: item-item
collaborative filtering, MMR diversity re-ranking, Hacker-News trending, and
time-decayed taste profiles.

No database. No server. No global cache. Every function takes its data as an
argument and returns a value — so the whole algorithm is testable in
isolation.

```text
  user_items: {user_id -> {item_ids}}
        │
        ├──► build_index_from_user_items()  ──► ItemCFIndex
        │         co-occurrence + normalised cosine
        │                  │
        │                  ├──► recommend_from_seen()  → candidates
        │                  └──► similar_items()         → "more like this"
        │
        ├──► build_profile()  ──► TasteProfile   (half-life decay)
        │
        └──► hn_score()      ──► trending       (age-discounted)

  candidates ──► filter_*()  ──► mmr_select()  ──► final list
                 hard filters     diversity
```

Open-source reference implementation extracted from my own open-source project
MIYU — a music streaming platform with an AI service (MIT).

## Install

```bash
pip install recsys-core
```

## Item-item collaborative filtering

```python
from recsys_core import build_index_from_user_items, recommend_from_seen

user_items = {
    1: {10, 20, 30},
    2: {10, 20, 30},
    3: {10, 20, 40},
}

index = build_index_from_user_items(user_items, min_cooccurrence=2)

recommend_from_seen([10], index)
# [(20, 0.91), (30, 0.86)]

[![CI](https://github.com/klimenkod406/recsys-core/actions/workflows/ci.yml/badge.svg)](https://github.com/klimenkod406/recsys-core/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
```

The algorithm:

1. **Co-occurrence** — `count(i, j)` is the number of users holding both `i`
   and `j`.
2. **Normalised cosine** — `sim(i, j) = co(i, j) / sqrt(c(i) * c(j))`, where
   `c(i)` is the number of users holding `i`. This is what stops a popular
   item from being everybody's neighbour.
3. **Top-N neighbours** per item, sorted by similarity.
4. **Sparse aggregation** — `score(j) = Σ sim(i, j)` over the user's own
   items, for `j` not yet seen.

`min_cooccurrence` (default 2) drops pairs fewer than that many users share.
One shared item is usually coincidence, not preference.

## Diversity re-ranking

A candidate list ordered purely by relevance is repetitive. MMR trades a
little relevance for variety:

```
score(c) = λ · relevance(c) − (1 − λ) · max similarity to already-selected
```

```python
import numpy as np
from recsys_core import mmr_select

embeddings = {
    1: np.array([1.0, 0.0]),
    2: np.array([0.99, 0.1]),   # nearly identical to 1
    3: np.array([0.0, 1.0]),   # different
}

mmr_select([(1, 0.9), (2, 0.9), (3, 0.85)], embeddings, limit=2, lambda_diversity=0.5)
# [(1, 0.9), (3, 0.85)]  — item 2 was skipped as a near-duplicate

[![CI](https://github.com/klimenkod406/recsys-core/actions/workflows/ci.yml/badge.svg)](https://github.com/klimenkod406/recsys-core/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
```

`λ = 1` is pure relevance, `λ = 0` is pure diversity; the default `0.7` keeps
relevance in charge while breaking up runs. `max_per_artist` caps how many
items one artist may contribute.

### Hard filters

Applied before diversity is considered:

```python
from recsys_core import filter_seen_items, filter_explicit_dislikes, filter_blocked_artists

candidates = filter_seen_items(candidates, seen_ids)
candidates = filter_explicit_dislikes(candidates, disliked_ids)
candidates = filter_blocked_artists(candidates, artist_of, blocked_artist_ids)
```

## Trending

The Hacker News ranking formula:

```python
from recsys_core import hn_score

hn_score(plays=100, hours_age=10)
# (100 - 1) / (10 + 2) ** 1.8

[![CI](https://github.com/klimenkod406/recsys-core/actions/workflows/ci.yml/badge.svg)](https://github.com/klimenkod406/recsys-core/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
```

Two details matter: `plays - 1` keeps an item with no plays at zero instead
of negative, and `+ 2` in the denominator stops a brand-new item from
dominating on a single play.

## Taste profiles

A weighted view of what a user likes, decaying with a half life:

```python
from recsys_core import build_profile

profile = build_profile([
    (10, "like",    now),
    (20, "playlist", now - timedelta(days=30)),
], now=now)

profile.item_weights
# {10: 3.0, 20: 1.0}   — the playlist add is half a month old

[![CI](https://github.com/klimenkod406/recsys-core/actions/workflows/ci.yml/badge.svg)](https://github.com/klimenkod406/recsys-core/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
```

`decay = 0.5 ** (days_old / HALF_LIFE_DAYS)`, so an event today is worth
1.0, one 30 days old is worth 0.5, one 120 days old is worth 0.06.

Signals are weighted by how deliberate they were — a like counts more than a
partial play, and an early skip counts against the profile:

| Signal | Weight |
|---|---|
| `like` | 3.0 |
| `playlist` | 2.0 |
| `follow_artist` | 1.5 |
| `completed_play` | 1.0 |
| `partial_play` | 0.3 |
| `explicit_dislike` | −1.5 |
| `early_skip` | −0.8 |

Override the table per deployment with `signal_weight(signal, weights)` or by
passing `weights=` to `build_profile`.

## Design notes

- **Pure functions only.** No database, no file cache, no module-level
  mutable state. The original implementation mixed the algorithm with a
  SQLite wrapper and a pickled snapshot; the algorithm was what was worth
  keeping, so only that was kept.
- **Deterministic.** The same input always produces the same output, which
  makes the algorithm reproducible and the tests meaningful.
- **Unicode-safe identifiers.** The type hints use `int` for item ids rather
  than being tied to one schema.
- **Unknown signals weigh zero** rather than raising, so a new signal type
  can be logged before it is modelled.
- **Future events never go negative** — clock skew cannot subtract from a
  profile.

## Testing

```bash
pip install -e ".[dev]"
pytest
pytest --cov=recsys_core --cov-report=term-missing
```

62 tests cover: index construction (symmetry, normalisation, truncation,
determinism), recommendation (exclusions, limits, ordering), MMR (the
relevance/diversity trade-off, artist caps, missing embeddings), the
trending formula, profile decay and signal weights, and profile similarity.

Coverage is 98%.

## License

MIT
