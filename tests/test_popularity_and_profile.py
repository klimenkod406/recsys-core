"""Tests for the trending score and the taste profile."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recsys_core import popularity, profile

NOW = datetime(2026, 11, 5, 12, 0, 0, tzinfo=timezone.utc)


class TestHnScore:
    def test_zero_plays_scores_zero(self):
        assert popularity.hn_score(0, 1.0) == 0.0

    def test_negative_plays_scores_zero(self):
        assert popularity.hn_score(-5, 1.0) == 0.0

    def test_new_item_with_one_play_scores_zero(self):
        # plays - 1 keeps a single play at zero rather than negative.
        assert popularity.hn_score(1, 1.0) == 0.0

    def test_known_formula(self):
        # (100 - 1) / (10 + 2) ** 1.8
        expected = 99 / (12**1.8)
        assert popularity.hn_score(100, 10.0) == pytest.approx(expected)

    def test_newer_item_outranks_older_one(self):
        fresh = popularity.hn_score(100, 1.0)
        old = popularity.hn_score(100, 100.0)
        assert fresh > old

    def test_more_plays_outranks_fewer(self):
        assert popularity.hn_score(100, 10.0) > popularity.hn_score(50, 10.0)

    def test_negative_age_is_clamped(self):
        assert popularity.hn_score(100, -5.0) == pytest.approx(
            popularity.hn_score(100, 0.0)
        )

    def test_alias_matches(self):
        assert popularity.trending_score(100, 10.0) == popularity.hn_score(100, 10.0)


class TestTimeDecay:
    def test_missing_timestamp_is_full_weight(self):
        assert profile.time_decay(None, NOW) == 1.0

    def test_event_now_is_full_weight(self):
        assert profile.time_decay(NOW, NOW) == pytest.approx(1.0)

    def test_one_half_life_halves_the_weight(self):
        event = NOW - timedelta(days=profile.HALF_LIFE_DAYS)
        assert profile.time_decay(event, NOW) == pytest.approx(0.5)

    def test_two_half_lives_quarter_the_weight(self):
        event = NOW - timedelta(days=2 * profile.HALF_LIFE_DAYS)
        assert profile.time_decay(event, NOW) == pytest.approx(0.25)

    def test_future_event_is_not_negative(self):
        event = NOW + timedelta(days=5)
        assert profile.time_decay(event, NOW) == pytest.approx(1.0)


class TestSignalWeight:
    def test_known_signal(self):
        assert profile.signal_weight("like") == profile.WEIGHT_LIKE

    def test_negative_signal(self):
        assert profile.signal_weight("explicit_dislike") < 0

    def test_unknown_signal_weighs_zero(self):
        assert profile.signal_weight("not_a_signal") == 0.0

    def test_weights_can_be_overridden(self):
        assert profile.signal_weight("like", {"like": 9.0}) == 9.0


class TestBuildProfile:
    def test_accumulates_weights_per_item(self):
        events = [(1, "like", NOW), (1, "like", NOW)]
        p = profile.build_profile(events, now=NOW)
        assert p.item_weights[1] == pytest.approx(2 * profile.WEIGHT_LIKE)

    def test_counts_signals(self):
        events = [(1, "like", NOW), (2, "playlist", NOW)]
        p = profile.build_profile(events, now=NOW)
        assert p.signal_counts == {"like": 1, "playlist": 1}

    def test_old_events_count_less(self):
        fresh = profile.build_profile([(1, "like", NOW)], now=NOW)
        stale = profile.build_profile(
            [(1, "like", NOW - timedelta(days=30))], now=NOW
        )
        assert stale.item_weights[1] < fresh.item_weights[1]

    def test_unknown_signal_adds_no_weight_but_is_counted(self):
        p = profile.build_profile([(1, "mystery", NOW)], now=NOW)
        assert 1 not in p.item_weights
        assert p.signal_counts["mystery"] == 1

    def test_top_items_ordered_by_weight(self):
        events = [(1, "partial_play", NOW), (2, "like", NOW)]
        p = profile.build_profile(events, now=NOW)
        assert p.top_items()[0][0] == 2

    def test_top_items_respects_limit(self):
        events = [(i, "like", NOW) for i in range(5)]
        p = profile.build_profile(events, now=NOW)
        assert len(p.top_items(limit=2)) == 2


class TestProfileSimilarity:
    def test_identical_profiles_are_similar(self):
        a = {1: 1.0, 2: 2.0}
        assert profile.profile_similarity(a, a) == pytest.approx(1.0)

    def test_disjoint_profiles_score_zero(self):
        assert profile.profile_similarity({1: 1.0}, {2: 1.0}) == 0.0

    def test_empty_profile_scores_zero(self):
        assert profile.profile_similarity({}, {1: 1.0}) == 0.0
        assert profile.profile_similarity({1: 1.0}, {}) == 0.0

    def test_partial_overlap_is_between_zero_and_one(self):
        score = profile.profile_similarity({1: 1.0, 2: 1.0}, {1: 1.0, 3: 1.0})
        assert 0.0 < score < 1.0
