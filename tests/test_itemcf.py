"""Tests for item-item collaborative filtering."""

from __future__ import annotations

import math

import pytest

from recsys_core import itemcf


class TestBuildIndex:
    def test_empty_input_yields_empty_index(self):
        idx = itemcf.build_index_from_user_items({})
        assert idx.size == 0
        assert idx.n_users == 0

    def test_single_user_produces_no_pairs(self):
        # One user cannot co-occur anything with anything.
        idx = itemcf.build_index_from_user_items({1: {10}}, min_cooccurrence=1)
        assert idx.size == 0
        assert idx.item_freq == {10: 1}

    def test_pair_below_min_cooccurrence_is_dropped(self):
        user_items = {1: {10, 20}, 2: {10, 30}}
        idx = itemcf.build_index_from_user_items(user_items, min_cooccurrence=2)
        assert idx.size == 0

    def test_cooccurrence_is_symmetric(self):
        # Three users with {A, B}: sim(A, B) = 3 / sqrt(3 * 3) = 1.0
        user_items = {1: {1, 2}, 2: {1, 2}, 3: {1, 2}}
        idx = itemcf.build_index_from_user_items(user_items, min_cooccurrence=1)
        assert (2, pytest.approx(1.0)) in idx.neighbors[1]
        assert (1, pytest.approx(1.0)) in idx.neighbors[2]
        assert idx.item_freq == {1: 3, 2: 3}

    def test_normalisation_penalises_popular_items(self):
        # Item 1 is popular, item 2 is rare: the shared count is divided by
        # sqrt(c_a * c_b), so the rare pair scores higher.
        user_items = {i: {1} for i in range(10)}
        user_items[0].add(2)
        user_items[1].add(2)
        user_items[2].add(3)
        user_items[3].add(3)

        idx = itemcf.build_index_from_user_items(user_items, min_cooccurrence=1)

        sim12 = dict(idx.neighbors.get(1, []))[2]
        assert sim12 == pytest.approx(2 / math.sqrt(10 * 2), rel=1e-6)

    def test_top_neighbours_are_truncated(self):
        # Five users each hold {0, i}, so item 0 has five neighbours.
        user_items = {i: {0, i} for i in range(1, 6)}
        idx = itemcf.build_index_from_user_items(
            user_items, top_neighbors=3, min_cooccurrence=1
        )
        assert len(idx.neighbors[0]) == 3

    def test_neighbours_are_sorted_by_similarity(self):
        user_items = {1: {1, 2, 3}, 2: {1, 2, 3}, 3: {1, 2}, 4: {1, 3}}
        idx = itemcf.build_index_from_user_items(user_items, min_cooccurrence=1)

        scores = [sim for _, sim in idx.neighbors[1]]
        assert scores == sorted(scores, reverse=True)

    def test_index_records_user_count(self):
        idx = itemcf.build_index_from_user_items(
            {1: {10, 20}, 2: {10, 20}, 3: {10, 30}}, min_cooccurrence=1
        )
        assert idx.n_users == 3

    def test_build_is_deterministic(self):
        user_items = {1: {1, 2, 3}, 2: {1, 2}, 3: {2, 3}}
        a = itemcf.build_index_from_user_items(user_items, min_cooccurrence=1)
        b = itemcf.build_index_from_user_items(user_items, min_cooccurrence=1)
        assert a.neighbors == b.neighbors


class TestRecommend:
    @pytest.fixture
    def index(self):
        user_items = {
            1: {10, 20},
            2: {10, 20},
            3: {10, 20},
            4: {30, 40},
        }
        return itemcf.build_index_from_user_items(user_items, min_cooccurrence=1)

    def test_recommends_neighbours_of_seen_items(self, index):
        out = itemcf.recommend_from_seen([10], index)
        assert [tid for tid, _ in out] == [20]

    def test_excludes_the_seen_items(self, index):
        out = itemcf.recommend_from_seen([10, 20], index)
        assert out == []

    def test_extra_exclusions_are_honoured(self, index):
        out = itemcf.recommend_from_seen([10], index, exclude=[20])
        assert out == []

    def test_respects_the_limit(self, index):
        out = itemcf.recommend_from_seen([10], index, limit=0)
        assert out == []

    def test_scores_are_sorted_descending(self):
        user_items = {
            1: {1, 2, 3},
            2: {1, 2, 3},
            3: {1, 2, 3},
            4: {1, 2},
        }
        index = itemcf.build_index_from_user_items(user_items, min_cooccurrence=1)
        out = itemcf.recommend_from_seen([1], index)
        scores = [score for _, score in out]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_seen_item_yields_nothing(self, index):
        assert itemcf.recommend_from_seen([9999], index) == []

    def test_empty_seen_yields_nothing(self, index):
        assert itemcf.recommend_from_seen([], index) == []


class TestSimilarItems:
    @pytest.fixture
    def index(self):
        user_items = {1: {10, 20}, 2: {10, 20}, 3: {10, 20}}
        return itemcf.build_index_from_user_items(user_items, min_cooccurrence=1)

    def test_returns_the_neighbours(self, index):
        out = itemcf.similar_items(10, index)
        assert [tid for tid, _ in out] == [20]

    def test_respects_the_limit(self, index):
        assert itemcf.similar_items(10, index, limit=0) == []

    def test_unknown_item_returns_empty(self, index):
        assert itemcf.similar_items(9999, index) == []
