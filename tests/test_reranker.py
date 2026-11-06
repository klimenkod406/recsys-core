"""Tests for MMR re-ranking and the hard filters."""

from __future__ import annotations

import numpy as np
import pytest

from recsys_core import reranker


@pytest.fixture
def embeddings():
    # Three distinct directions.
    return {
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([0.99, 0.1, 0.0]),  # nearly identical to 1
        3: np.array([0.0, 1.0, 0.0]),  # orthogonal to both
    }


class TestCosineSim:
    def test_identical_vectors_are_similar(self):
        a = np.array([1.0, 2.0, 3.0])
        assert reranker.cosine_sim(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_dissimilar(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert reranker.cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_does_not_divide_by_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert reranker.cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)


class TestMMRSelect:
    def test_empty_candidates_return_empty(self, embeddings):
        assert reranker.mmr_select([], embeddings, limit=5) == []

    def test_lambda_one_is_pure_relevance(self, embeddings):
        candidates = [(1, 0.9), (2, 0.8), (3, 0.7)]
        out = reranker.mmr_select(
            candidates, embeddings, limit=3, lambda_diversity=1.0
        )
        assert [tid for tid, _ in out] == [1, 2, 3]

    def test_diversity_breaks_up_near_duplicates(self, embeddings):
        # Items 1 and 2 are nearly identical; 3 is not. Pure relevance would
        # pick 1 then 2; diversity should prefer 3 second.
        candidates = [(1, 0.9), (2, 0.9), (3, 0.85)]
        out = reranker.mmr_select(
            candidates, embeddings, limit=2, lambda_diversity=0.5
        )
        assert out[0][0] == 1
        assert out[1][0] == 3

    def test_limit_is_respected(self, embeddings):
        candidates = [(1, 0.9), (2, 0.8), (3, 0.7)]
        out = reranker.mmr_select(candidates, embeddings, limit=2)
        assert len(out) == 2

    def test_items_without_embeddings_get_no_penalty(self):
        candidates = [(1, 0.5), (2, 0.4)]
        out = reranker.mmr_select(candidates, {}, limit=2)
        assert [tid for tid, _ in out] == [1, 2]

    def test_artist_cap_is_enforced(self, embeddings):
        candidates = [(1, 0.9), (2, 0.8), (3, 0.7)]
        artist_of = {1: 100, 2: 100, 3: 200}
        out = reranker.mmr_select(
            candidates,
            embeddings,
            limit=3,
            max_per_artist=1,
            artist_of=artist_of,
        )
        artists = [artist_of[tid] for tid, _ in out]
        assert artists.count(100) == 1

    def test_artist_cap_none_allows_everything(self, embeddings):
        candidates = [(1, 0.9), (2, 0.8)]
        artist_of = {1: 100, 2: 100}
        out = reranker.mmr_select(
            candidates,
            embeddings,
            limit=2,
            max_per_artist=None,
            artist_of=artist_of,
        )
        assert len(out) == 2


class TestFilters:
    def test_filters_listened_today(self):
        candidates = [(1, 0.9), (2, 0.8), (3, 0.7)]
        out = reranker.filter_listened_today(candidates, [2])
        assert [tid for tid, _ in out] == [1, 3]

    def test_filters_explicit_dislikes(self):
        candidates = [(1, 0.9), (2, 0.8)]
        out = reranker.filter_explicit_dislikes(candidates, [1])
        assert [tid for tid, _ in out] == [2]

    def test_filters_seen_items(self):
        candidates = [(1, 0.9), (2, 0.8)]
        out = reranker.filter_seen_items(candidates, [1])
        assert [tid for tid, _ in out] == [2]

    def test_filters_blocked_artists(self):
        candidates = [(1, 0.9), (2, 0.8), (3, 0.7)]
        artist_of = {1: 100, 2: 200, 3: 300}
        out = reranker.filter_blocked_artists(candidates, artist_of, [200])
        assert [tid for tid, _ in out] == [1, 3]

    def test_items_without_an_artist_are_kept(self):
        candidates = [(1, 0.9)]
        out = reranker.filter_blocked_artists(candidates, {}, [200])
        assert [tid for tid, _ in out] == [1]

    def test_filters_accept_an_empty_exclusion_set(self):
        candidates = [(1, 0.9)]
        out = reranker.filter_seen_items(candidates, [])
        assert [tid for tid, _ in out] == [1]
