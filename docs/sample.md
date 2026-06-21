# Sample Knowledge Base

This is a comprehensive knowledge base for the RAGforApphelix project, a simple Retrieval-Augmented Generation system.

## Company Information

- **Project Name**: RAGforApphelix
- **Support Email**: apphelixblr@gmail.com
- **Phone Support**: +91-80-XXXX-XXXX
- **Office Location**: Bangalore, India
- **Office Hours**: 10:00 to 18:00 IST, Monday to Friday

## Core Features

### Vector Embeddings
RAGforApphelix uses advanced embedding models to convert documents into semantic vectors. This allows the system to understand the meaning of text, not just keywords. The embeddings are powered by models like BAAI/bge-small-en-v1.5, which provide excellent semantic understanding for English documents.

### Weaviate Vector Storage
All document chunks are stored in Weaviate, a vector database that supports semantic similarity search. Weaviate can run locally or remotely and stores both text and source metadata for efficient retrieval.

### BM25 Keyword Retrieval
For users who prefer traditional keyword-based search, RAGforApphelix offers BM25 retrieval mode. This uses the Okapi BM25 algorithm, which ranks documents based on keyword relevance scores.

### Groq LLM Integration
The system integrates with Groq's fast inference LLMs. Supported models include:
- llama-3.1-8b-instant (default)
- llama-3.1-70b-versatile
- mixtral-8x7b-32768
- Other Groq-supported models

## Getting Started

1. **Install Dependencies**: Run `pip install -r requirements.txt` to install Flask, Groq SDK, ChromaDB, fastembed, rank-bm25, and python-dotenv.

2. **Prepare Documents**: Place your .txt or .md files in the `docs/` folder. The system will automatically process and index them.

2.1 **Start Weaviate**: Run `docker compose up -d` from the project root to start a local Weaviate server on `http://localhost:8080`.

3. **Set Weaviate auth**: If your Weaviate instance does not require auth, set `RAG_WEAVIATE_AUTH_TYPE=none`. If it uses API key auth, set `RAG_WEAVIATE_AUTH_TYPE=api_key` and `RAG_WEAVIATE_API_KEY=...`. For bearer auth, set `RAG_WEAVIATE_AUTH_TYPE=bearer` and `RAG_WEAVIATE_BEARER_TOKEN=...`.

4. **Set API Key**: Either set the `GROQ_API_KEY` environment variable or provide it through the web interface.

4. **Configure Embeddings**: Set `RAG_EMBED_MODEL` to your preferred embedding model (default: BAAI/bge-small-en-v1.5).

5. **Run the Application**: Execute `python web_app.py` to start the Flask server on http://127.0.0.1:7860.

## Configuration Options

- `RAG_RETRIEVER`: Choose between "vector" (default) or "bm25" retrieval modes
- `RAG_DOCS_DIR`: Path to your documents directory (default: "docs")
- `RAG_WEAVIATE_URL`: Weaviate server URL (default: "http://localhost:8080")
- `RAG_WEAVIATE_AUTH_TYPE`: Authentication method for Weaviate: `none`, `api_key`, or `bearer` (default: `none`)
- `RAG_WEAVIATE_API_KEY`: Weaviate API key when using `api_key` auth
- `RAG_WEAVIATE_BEARER_TOKEN`: Bearer token when using `bearer` auth
- `RAG_CHUNK_CHARS`: Document chunk size in characters (default: 1200)
- `RAG_CHUNK_OVERLAP`: Overlap between chunks in characters (default: 200)
- `GROQ_MODEL`: LLM model to use (default: llama-3.1-8b-instant)
- `RAG_TOP_K`: Number of documents to retrieve (default: 4)

## Performance Metrics

- **Vector Search Latency**: < 100ms for queries on 10k+ documents
- **Embedding Generation**: ~50 documents/second on standard hardware
- **ChromaDB Index Size**: ~1MB per 1000 documents (depends on embedding dimension)
- **LLM Response Time**: 1-5 seconds depending on model size and context length

## Common Use Cases

### Customer Support Chatbots
Deploy RAGforApphelix to handle frequently asked questions automatically. The system can search through documentation and FAQs to provide accurate answers without human intervention.

### Internal Knowledge Management
Organizations can use RAGforApphelix to build searchable knowledge bases from wikis, documentation, and internal guides, making information easily accessible to employees.

### Research and Document Analysis
Researchers can leverage RAGforApphelix to quickly search through academic papers, reports, and datasets to find relevant information for their projects.

### Product Documentation
Product teams can create intelligent search for their documentation, allowing users to find answers through natural language queries instead of navigating complex menu structures.

## Supported Document Formats

- **Markdown** (.md): Full markdown syntax support with headers, lists, code blocks, etc.
- **Plain Text** (.txt): Simple text files with automatic paragraph detection

## Troubleshooting

### Empty Response from Model
If the LLM returns an empty response, try:
- Rephrasing your question more simply
- Switching to a different model like llama-3.1-70b-versatile
- Checking that your documents contain relevant information
- Verifying your GROQ_API_KEY is valid

### Slow Search Performance
Improve search speed by:
- Reducing chunk size (RAG_CHUNK_CHARS)
- Decreasing top_k value
- Using BM25 mode for faster keyword-based retrieval
- Adding more RAM for better ChromaDB caching

### Index Not Updating
If new documents don't appear in search results:
- Clear the `.chroma_db/` directory to force re-indexing
- Check that files are in the correct `docs/` directory
- Verify file extensions are .txt or .md
- Ensure documents contain actual content

## Security Considerations

- **API Key Protection**: Never commit GROQ_API_KEY to version control. Use environment variables or .env files.
- **Rate Limiting**: Monitor your Groq API usage to avoid unexpected costs.
- **Document Privacy**: Keep sensitive documents in private deployment environments.
- **Access Control**: Consider adding authentication layer when deploying to production.

## API Examples

### Python CLI Usage
```
python rag_groq.py "What is the company email?"
```

### Environment Variables
```
GROQ_API_KEY=your_key_here
RAG_RETRIEVER=vector
GROQ_MODEL=llama-3.1-8b-instant
```

## Support Resources

- **Email Support**: apphelixblr@gmail.com
- **Response Time**: 24-48 hours during business hours
- **Documentation**: Check README.md and code comments for detailed information
- **GitHub Issues**: Report bugs and request features through GitHub (if applicable)

