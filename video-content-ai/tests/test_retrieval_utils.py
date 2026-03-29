"""Unit tests for sentence retrieval helpers (pure logic + mocked embeddings)."""

from unittest.mock import patch

import numpy as np

from app.codexvid.retrieval_utils import (
    cosine_similarity_matrix,
    embed_texts,
    filter_sentences_overlapping_chunks,
    find_most_relevant_sentence,
)


def test_cosine_similarity_matrix():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    m = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    sims = cosine_similarity_matrix(q, m)
    assert sims.shape == (2,)
    assert sims[1] > sims[0]


def test_filter_sentences_overlapping_chunks():
    sents = [
        {"text": "a", "start": 1.0, "end": 2.0},
        {"text": "b", "start": 50.0, "end": 51.0},
    ]
    chunks = [{"text": "x", "start_time": 0.0, "end_time": 5.0, "start": 0.0, "end": 5.0}]
    out = filter_sentences_overlapping_chunks(sents, chunks)
    assert len(out) == 1
    assert out[0]["text"] == "a"


@patch("app.codexvid.retrieval_utils.embed_texts")
def test_find_most_relevant_sentence_batches_once(mock_embed):
    mock_embed.return_value = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    sents = [
        {"text": "alpha", "start": 0.0, "end": 1.0},
        {"text": "beta query match", "start": 2.0, "end": 3.0},
    ]
    best = find_most_relevant_sentence("query match", sents, embed_model="dummy")
    assert best is not None
    assert best["text"] == "beta query match"
    mock_embed.assert_called_once()
    call_texts = mock_embed.call_args[0][0]
    assert call_texts[0] == "query match"
    assert len(call_texts) == 3


def test_snap_chapter_times_to_sentences():
    from app.codexvid.teaching import snap_chapter_times_to_sentences

    sents = [
        {"text": "one", "start": 10.0, "end": 12.0},
        {"text": "two", "start": 20.0, "end": 25.0},
    ]
    chapters = [{"title": "A", "start": 10.4, "end": 24.0, "explanation": ""}]
    out = snap_chapter_times_to_sentences(chapters, sents)
    assert len(out) == 1
    assert out[0]["start"] == 10.0
    assert out[0]["end"] == 25.0
