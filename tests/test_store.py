"""Tests for resonant_archive.store — Chroma integration with namespaces.

These tests exercise the real sentence-transformers model and a real
Chroma PersistentClient in a temporary directory. The embedder is shared
across tests via a module-scoped fixture because model loading is slow.

Copyright (C) 2025-2026 Codependent AI
Licensed under the Codependent AI Source-Available License. See LICENSE.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from resonant_archive.chunker import Chunk
from resonant_archive.embed import Embedder
from resonant_archive.store import ArchiveStore, SearchResult


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """One embedder for the whole module — model load is expensive."""
    return Embedder()


@pytest.fixture
def tmp_store() -> Iterator[ArchiveStore]:
    # ``ignore_cleanup_errors`` is required on Windows because Chroma's
    # PersistentClient keeps file handles open for the lifetime of the
    # Python process. Tests pass regardless; the temp dir is cleaned up
    # when pytest exits.
    with tempfile.TemporaryDirectory(
        prefix="resonant-archive-store-", ignore_cleanup_errors=True
    ) as d:
        yield ArchiveStore(data_dir=Path(d))


def _make_chunks(texts: list[str], source: str = "test.md") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{source}-{i}",
            source_file=source,
            text=text,
            chunk_index=i,
            char_offset=i * 100,
            title="Test Note",
        )
        for i, text in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_empty_store_count_is_zero(tmp_store: ArchiveStore) -> None:
    assert tmp_store.count() == 0
    assert tmp_store.list_namespaces() == {}


def test_add_chunks_and_count(
    tmp_store: ArchiveStore, embedder: Embedder
) -> None:
    chunks = _make_chunks(
        ["hello world", "the quick brown fox", "resonant archive test"]
    )
    added = tmp_store.add_chunks(chunks, namespace="test-ns", embedder=embedder)
    assert added == 3
    assert tmp_store.count() == 3
    assert tmp_store.count(namespace="test-ns") == 3
    assert tmp_store.count(namespace="other-ns") == 0


def test_empty_chunks_is_noop(
    tmp_store: ArchiveStore, embedder: Embedder
) -> None:
    added = tmp_store.add_chunks([], namespace="ns", embedder=embedder)
    assert added == 0
    assert tmp_store.count() == 0


def test_empty_namespace_raises(
    tmp_store: ArchiveStore, embedder: Embedder
) -> None:
    chunks = _make_chunks(["hello"])
    with pytest.raises(ValueError):
        tmp_store.add_chunks(chunks, namespace="", embedder=embedder)


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


def test_search_returns_relevant_results(
    tmp_store: ArchiveStore, embedder: Embedder
) -> None:
    chunks = _make_chunks(
        [
            "A cat sleeps on the mat.",
            "The kitten is napping on the rug.",
            "A sports car speeds down the highway.",
        ]
    )
    tmp_store.add_chunks(chunks, namespace="animals", embedder=embedder)
    results = tmp_store.search("feline nap", embedder, n_results=3)
    assert len(results) == 3
    # The two most similar results should be cat/kitten, not the car.
    top_two_texts = " ".join(r.text.lower() for r in results[:2])
    assert "cat" in top_two_texts or "kitten" in top_two_texts


def test_search_empty_query_returns_nothing(
    tmp_store: ArchiveStore, embedder: Embedder
) -> None:
    tmp_store.add_chunks(
        _make_chunks(["a chunk"]), namespace="ns", embedder=embedder
    )
    assert tmp_store.search("", embedder) == []
    assert tmp_store.search("   ", embedder) == []


def test_search_returns_provenance_metadata(
    tmp_store: ArchiveStore, embedder: Embedder
) -> None:
    chunks = [
        Chunk(
            chunk_id="conv-1-0",
            source_file="chatgpt/conv-1.md",
            text="A conversation about Python pytest patterns",
            chunk_index=0,
            char_offset=0,
            title="Python Testing",
            timestamp="2024-03-15T10:30:00+00:00",
            timestamp_source="frontmatter",
            conversation_id="conv-123",
            provider="chatgpt",
        )
    ]
    tmp_store.add_chunks(chunks, namespace="chat-archive", embedder=embedder)
    results = tmp_store.search("pytest", embedder, n_results=1)
    assert len(results) == 1
    r: SearchResult = results[0]
    assert r.namespace == "chat-archive"
    assert r.title == "Python Testing"
    assert r.timestamp == "2024-03-15T10:30:00+00:00"
    assert r.conversation_id == "conv-123"
    assert r.provider == "chatgpt"
    assert r.source_file == "chatgpt/conv-1.md"
    assert 0.0 <= r.relevance <= 1.0


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------


def test_namespace_filter_isolates_results(
    tmp_store: ArchiveStore, embedder: Embedder
) -> None:
    tmp_store.add_chunks(
        _make_chunks(["apple pie recipe"], "recipes.md"),
        namespace="cooking",
        embedder=embedder,
    )
    tmp_store.add_chunks(
        _make_chunks(["apple tree in the garden"], "garden.md"),
        namespace="outdoors",
        embedder=embedder,
    )
    assert tmp_store.count() == 2

    cooking = tmp_store.search(
        "apple", embedder, n_results=5, namespace="cooking"
    )
    assert len(cooking) == 1
    assert cooking[0].namespace == "cooking"
    assert "pie" in cooking[0].text

    outdoors = tmp_store.search(
        "apple", embedder, n_results=5, namespace="outdoors"
    )
    assert len(outdoors) == 1
    assert outdoors[0].namespace == "outdoors"
    assert "tree" in outdoors[0].text


def test_unfiltered_search_sees_all_namespaces(
    tmp_store: ArchiveStore, embedder: Embedder
) -> None:
    tmp_store.add_chunks(
        _make_chunks(["apple pie recipe"], "recipes.md"),
        namespace="cooking",
        embedder=embedder,
    )
    tmp_store.add_chunks(
        _make_chunks(["apple tree in the garden"], "garden.md"),
        namespace="outdoors",
        embedder=embedder,
    )
    results = tmp_store.search("apple", embedder, n_results=5)
    assert len(results) == 2
    assert {r.namespace for r in results} == {"cooking", "outdoors"}


def test_list_namespaces(
    tmp_store: ArchiveStore, embedder: Embedder
) -> None:
    tmp_store.add_chunks(
        _make_chunks(["a", "b"], "one.md"),
        namespace="ns-a",
        embedder=embedder,
    )
    tmp_store.add_chunks(
        _make_chunks(["c"], "two.md"),
        namespace="ns-b",
        embedder=embedder,
    )
    ns_counts = tmp_store.list_namespaces()
    assert ns_counts.get("ns-a") == 2
    assert ns_counts.get("ns-b") == 1


def test_list_namespaces_paginates(
    tmp_store: ArchiveStore, embedder: Embedder
) -> None:
    tmp_store.add_chunks(
        _make_chunks(["a", "b", "c"], "one.md"),
        namespace="ns-a",
        embedder=embedder,
    )
    tmp_store.add_chunks(
        _make_chunks(["d", "e"], "two.md"),
        namespace="ns-b",
        embedder=embedder,
    )
    expected = {"ns-a": 3, "ns-b": 2}
    # Page smaller than the collection: counts survive the walk.
    assert tmp_store.list_namespaces(batch_size=2) == expected
    # Page size divides the collection exactly: the trailing empty page
    # must end the walk without double-counting.
    assert tmp_store.list_namespaces(batch_size=5) == expected
    # Default page size: unchanged behaviour on small collections.
    assert tmp_store.list_namespaces() == expected


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_upsert_is_idempotent(
    tmp_store: ArchiveStore, embedder: Embedder
) -> None:
    chunks = _make_chunks(["hello", "world"])
    tmp_store.add_chunks(chunks, namespace="ns", embedder=embedder)
    tmp_store.add_chunks(chunks, namespace="ns", embedder=embedder)
    # Re-adding the same chunk_ids does not duplicate rows.
    assert tmp_store.count() == 2
