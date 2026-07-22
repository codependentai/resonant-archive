"""resonant-archive store — Chroma-backed persistent archive with namespaces.

The store keeps everything in a single Chroma collection and uses a
``namespace`` metadata field to separate different corpora. This gives us:

  - **Namespace-scoped search** via Chroma's metadata filter
  - **Cross-namespace search** by simply omitting the filter
  - **One embedding model + one index** shared across all namespaces
  - **Provenance in every result** so querying AIs see which corpus the
    hit came from ("chatgpt-2024", "journals", etc.)

Copyright (C) 2025-2026 Codependent AI
Licensed under the Codependent AI Source-Available License. See LICENSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb

from .chunker import Chunk
from .embed import Embedder

#: Collection name used inside Chroma. Users don't normally see this —
#: they interact via namespaces instead.
DEFAULT_COLLECTION_NAME = "resonant_archive"


@dataclass
class SearchResult:
    """A single search hit returned by ``ArchiveStore.search``.

    ``relevance`` is a 0..1 score where higher is better (derived from
    Chroma's cosine distance via ``1 - distance / 2``).
    """

    chunk_id: str
    source_file: str
    text: str
    relevance: float
    namespace: str | None = None
    title: str | None = None
    timestamp: str | None = None
    conversation_id: str | None = None
    provider: str | None = None
    chunk_index: int = 0


class ArchiveStore:
    """Persistent Chroma archive with namespace tagging.

    Typical usage::

        store = ArchiveStore(data_dir=Path.home() / ".resonant-archive")
        embedder = Embedder()
        store.add_chunks(chunks, namespace="chatgpt-2024", embedder=embedder)
        hits = store.search("identity formation", embedder, namespace="chatgpt-2024")
    """

    def __init__(
        self,
        data_dir: Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.data_dir = data_dir
        self.chroma_dir = data_dir / "chroma"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Resonant Archive semantic index"},
        )

    # -- counts & introspection ---------------------------------------------

    def count(self, namespace: str | None = None) -> int:
        """Return total indexed chunks, optionally filtered by namespace."""
        if namespace is None:
            return self._collection.count()
        result = self._collection.get(
            where={"namespace": namespace}, include=[]
        )
        return len(result.get("ids") or [])

    def list_namespaces(self, batch_size: int = 2000) -> dict[str, int]:
        """Return a ``{namespace: chunk_count}`` mapping for all namespaces.

        Walks the full collection metadata in batches to avoid SQLite
        variable limits on large indexes. Namespace bookkeeping is a v2
        optimization if needed.
        """
        counts: dict[str, int] = {}
        offset = 0
        while True:
            result = self._collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            metadatas = result.get("metadatas") or []
            if not metadatas:
                break
            for meta in metadatas:
                if not meta:
                    continue
                ns = meta.get("namespace", "<unnamed>")
                counts[ns] = counts.get(ns, 0) + 1
            if len(metadatas) < batch_size:
                break
            offset += batch_size
        return counts

    # -- writes --------------------------------------------------------------

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        namespace: str,
        embedder: Embedder,
        batch_size: int = 64,
    ) -> int:
        """Embed chunks and upsert them into the store.

        Chunks are keyed by ``chunk_id``, so re-adding the same chunks is
        idempotent — any existing entry with the same id is overwritten.
        Returns the total number of chunks written.
        """
        if not chunks:
            return 0
        if not namespace:
            raise ValueError("namespace must be a non-empty string")

        total = 0
        chunks_list = list(chunks)
        for start in range(0, len(chunks_list), batch_size):
            batch = chunks_list[start : start + batch_size]
            texts = [c.text for c in batch]
            vectors = embedder.embed(texts)
            ids = [c.chunk_id for c in batch]
            metadatas = [_chunk_metadata(c, namespace) for c in batch]
            self._collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )
            total += len(batch)
        return total

    # -- reads ---------------------------------------------------------------

    def search(
        self,
        query: str,
        embedder: Embedder,
        n_results: int = 5,
        namespace: str | None = None,
    ) -> list[SearchResult]:
        """Semantic search across the archive.

        If ``namespace`` is supplied, results are limited to that namespace.
        Otherwise the search spans all namespaces and each result is tagged
        with its origin namespace.
        """
        if not query or not query.strip():
            return []
        query_vec = embedder.embed([query])[0]
        where = {"namespace": namespace} if namespace else None
        raw = self._collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _format_results(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_metadata(chunk: Chunk, namespace: str) -> dict[str, Any]:
    """Build the Chroma metadata dict for a chunk.

    Chroma metadata values must be primitives (str / int / float / bool).
    Lists like ``tags`` are joined as comma-separated strings so they
    survive the round-trip.
    """
    meta: dict[str, Any] = {
        "namespace": namespace,
        "source_file": chunk.source_file,
        "chunk_index": chunk.chunk_index,
        "char_offset": chunk.char_offset,
    }
    if chunk.title is not None:
        meta["title"] = chunk.title
    if chunk.timestamp is not None:
        meta["timestamp"] = chunk.timestamp
    if chunk.timestamp_source is not None:
        meta["timestamp_source"] = chunk.timestamp_source
    if chunk.tags:
        meta["tags"] = ",".join(chunk.tags)
    if chunk.conversation_id is not None:
        meta["conversation_id"] = chunk.conversation_id
    if chunk.provider is not None:
        meta["provider"] = chunk.provider
    return meta


def _format_results(raw: dict[str, Any]) -> list[SearchResult]:
    """Convert a raw Chroma query response into ``SearchResult`` objects."""
    ids_lists = raw.get("ids") or [[]]
    if not ids_lists or not ids_lists[0]:
        return []

    ids = ids_lists[0]
    docs_lists = raw.get("documents") or [[]]
    meta_lists = raw.get("metadatas") or [[]]
    dist_lists = raw.get("distances") or [[]]
    docs = docs_lists[0] if docs_lists else []
    metas = meta_lists[0] if meta_lists else []
    dists = dist_lists[0] if dist_lists else []

    out: list[SearchResult] = []
    for i, chunk_id in enumerate(ids):
        meta: dict[str, Any] = metas[i] if i < len(metas) and metas[i] else {}
        doc = docs[i] if i < len(docs) else ""
        distance = dists[i] if i < len(dists) else 2.0
        # Chroma returns cosine distance in [0, 2]. Normalize to a relevance
        # score in [0, 1] where higher = more similar.
        relevance = max(0.0, 1.0 - float(distance) / 2.0)
        out.append(
            SearchResult(
                chunk_id=str(chunk_id),
                source_file=str(meta.get("source_file", "unknown")),
                text=str(doc) if doc is not None else "",
                relevance=relevance,
                namespace=meta.get("namespace"),
                title=meta.get("title"),
                timestamp=meta.get("timestamp"),
                conversation_id=meta.get("conversation_id"),
                provider=meta.get("provider"),
                chunk_index=int(meta.get("chunk_index", 0)),
            )
        )
    return out
