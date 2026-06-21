import sys
import pathlib
from langchain_core.tools import tool

# Ensure roots are in path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project.langchain.retriever import CustomContextRetriever
from context_pipeline.token_budget import count_tokens
from context_pipeline.compression import compress_chunks
from RAG_groq import Chunk, format_context

@tool
def retrieve_context(query: str, docs_dir: str = "docs", retriever_mode: str = "vector", top_k: int = 4) -> str:
    """Retrieve compressed context relevant to the user query from the documents directory."""
    retriever = CustomContextRetriever(docs_dir=docs_dir, retriever_mode=retriever_mode, top_k=top_k)
    docs = retriever.invoke(query)
    # Format documents as string
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[{i}] source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

@tool
def calculate_tokens(text: str, model: str = "llama-3.1-8b-instant") -> int:
    """Calculate the number of tokens in the given text using tiktoken/heuristics."""
    return count_tokens(text, model)

@tool
def compress_context(query: str, text: str) -> str:
    """Compress context text using extractive sentences based on the query."""
    # Convert text to standard Chunk format expected by compress_chunks
    chunks = [Chunk(source="ad-hoc", text=text)]
    from context_pipeline.config import PipelineConfig
    cfg = PipelineConfig.from_env()
    compressed = compress_chunks(query, chunks, cfg)
    return format_context(compressed)
