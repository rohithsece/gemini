"""
Vector Store — Weaviate v4 + FastEmbed
Optimized for Windows via Docker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import weaviate
import weaviate.classes.config as wc
from fastembed import TextEmbedding

from RAG_groq import Chunk, build_chunks, fingerprint_docs_dir

COLLECTION_NAME = "RAGChunk"
META_FILE = ".weaviate_meta.json"

# Module-level persistent client
_client: weaviate.WeaviateClient | None = None
_cache: dict[tuple[str, str], "VectorRetriever"] = {}


def _get_client() -> weaviate.WeaviateClient:
    """Connect to Weaviate (Local or Cloud)."""
    global _client
    if _client is not None:
        try:
            if _client.is_ready():
                return _client
        except Exception:
            pass
        _client = None

    url = os.environ.get("RAG_WEAVIATE_URL", "http://localhost:8080").strip()
    auth_type = os.environ.get("RAG_WEAVIATE_AUTH_TYPE", "none").strip().lower()
    api_key = os.environ.get("RAG_WEAVIATE_API_KEY", "").strip()
    bearer_token = os.environ.get("RAG_WEAVIATE_BEARER_TOKEN", "").strip()

    auth = None
    if auth_type == "api_key" and api_key:
        auth = weaviate.auth.AuthApiKey(api_key)
    elif auth_type == "bearer" and bearer_token:
        auth = weaviate.auth.AuthBearerToken(bearer_token)

    # If it's a cloud URL, use connect_to_weaviate_cloud
    if "weaviate.network" in url or "weaviate.cloud" in url:
        # Extract cluster name from URL (e.g., https://my-rag.weaviate.network -> my-rag)
        cluster_url = url.replace("https://", "").replace("http://", "")
        if auth:
            _client = weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=auth,
            )
        else:
            _client = weaviate.connect_to_weaviate_cloud(cluster_url=cluster_url) # type: ignore
    else:
        # Try local Docker first; if that fails, use BM25 fallback in app.
        try:
            if auth:
                _client = weaviate.connect_to_local(
                    host="localhost",
                    port=8080,
                    grpc_port=50051,
                    auth_credentials=auth,
                )
            else:
                _client = weaviate.connect_to_local(
                    host="localhost",
                    port=8080,
                    grpc_port=50051,
                )
        except Exception:
            raise
    return _client # type: ignore


def check_weaviate_ready() -> bool:
    """Helper for web_app status chip."""
    try:
        client = _get_client()
        return client.is_ready()
    except Exception:
        return False


def _collection_exists(client: weaviate.WeaviateClient) -> bool:
    return client.collections.exists(COLLECTION_NAME)


def _create_collection(client: weaviate.WeaviateClient) -> None:
    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=wc.Configure.Vectorizer.none(),
        properties=[
            wc.Property(
                name="text",
                data_type=wc.DataType.TEXT,
                tokenization=wc.Tokenization.WORD
            ),
            wc.Property(
                name="source",
                data_type=wc.DataType.TEXT
            ),
        ],
    )


class VectorRetriever:
    """Retrieval via Weaviate v4 Hybrid Search."""

    def __init__(self, client: weaviate.WeaviateClient, embedder: TextEmbedding):
        self._client = client
        self._embedder = embedder

    @property
    def embedder(self) -> TextEmbedding:
        """Expose embedder for downstream reranking (same model as indexing)."""
        return self._embedder

    def search(self, query: str, *, k: int = 4) -> list[Chunk]:
        """Return top-k chunks (same as hybrid oversampled with limit=k)."""
        alpha = float(os.environ.get("RAG_HYBRID_ALPHA", "0.75"))
        return self.search_hybrid_oversampled(query, candidate_limit=max(1, k), alpha=alpha)[:k]

    def search_hybrid_oversampled(
        self,
        query: str,
        *,
        candidate_limit: int,
        alpha: float | None = None,
    ) -> list[Chunk]:
        """
        Hybrid search (dense vector + sparse BM25 inside Weaviate).

        Weaviate's hybrid query fuses keyword (inverted index) and vector similarity
        in one ranked list. The ``alpha`` parameter controls the blend:
        - alpha near 1.0 → emphasize dense (semantic) retrieval
        - alpha near 0.0 → emphasize BM25 (lexical) retrieval
        Production systems tune alpha per domain (e.g. legal search often leans lexical).

        Oversampling (candidate_limit > final top_k) gives downstream stages
        (dedupe, rerank, compression) more material to work with — a common pattern.
        """
        lim = max(1, candidate_limit)
        if alpha is None:
            alpha = float(os.environ.get("RAG_HYBRID_ALPHA", "0.75"))
        alpha = max(0.0, min(1.0, alpha))

        qvec = list(self._embedder.embed([query]))
        if not qvec:
            return []
        v = qvec[0].tolist() if hasattr(qvec[0], "tolist") else list(qvec[0])

        col = self._client.collections.get(COLLECTION_NAME)
        results = col.query.hybrid(
            query=query,
            vector=v,
            limit=lim,
            alpha=alpha,
            return_properties=["text", "source"],
        )

        out: list[Chunk] = []
        for obj in results.objects:
            text = obj.properties.get("text")
            src = obj.properties.get("source") or "unknown"
            if text:
                out.append(Chunk(source=str(src), text=str(text)))
        return out

    @classmethod
    def from_docs(cls, docs_dir: Path) -> "VectorRetriever":
        embed_name = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        fp = _index_fingerprint(docs_dir)
        cache_key = (str(docs_dir.resolve()), embed_name)

        client = _get_client()
        meta_file = Path(META_FILE).resolve()
        
        # Load meta to see if we need to re-index
        meta = None
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except: pass

        chunks = build_chunks(docs_dir)
        
        # Check if already indexed
        if _collection_exists(client) and meta and meta.get("fp") == fp:
            embedder = TextEmbedding(model_name=embed_name)
            return cls(client, embedder)

        # Rebuild Index
        if _collection_exists(client):
            client.collections.delete(COLLECTION_NAME)
        _create_collection(client)

        embedder = TextEmbedding(model_name=embed_name)
        col = client.collections.get(COLLECTION_NAME)

        with col.batch.fixed_size(batch_size=64) as batch:
            for chunk in chunks:
                vec = list(embedder.embed([chunk.text]))[0].tolist()
                batch.add_object(
                    properties={"text": chunk.text, "source": chunk.source},
                    vector=vec
                )

        meta_file.write_text(json.dumps({"fp": fp}))
        return cls(client, embedder)

def _index_fingerprint(docs_dir: Path) -> str:
    return fingerprint_docs_dir(docs_dir)
