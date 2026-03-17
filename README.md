# Production-Grade RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, Mistral API, and ChromaDB. This system enables intelligent document ingestion, semantic search, and context-aware question answering.

## Overview

The RAG Pipeline provides a production-ready solution for building intelligent applications that can ingest documents, generate embeddings, store them in a vector database, and answer user questions based on the ingested content. The pipeline leverages the Mistral API for both embedding generation (using the codestral-embed model) and large language model inference (using mistral-large-latest).

## Features

The pipeline includes the following core capabilities:

**Document Ingestion**: The system supports PDF and Word (.docx) file uploads. Documents are automatically processed, split into manageable chunks, and embedded using Mistral's codestral-embed model. Embeddings are stored in ChromaDB for efficient similarity-based retrieval.

**Intelligent Querying**: Users can submit questions that are embedded and matched against stored document chunks. The system retrieves the most relevant chunks and uses Mistral's language model to generate contextual answers based on the retrieved information.

**Vector Storage**: ChromaDB provides efficient vector storage with cosine similarity search, enabling fast retrieval of relevant document chunks from large document collections.

**Health Monitoring**: A dedicated health check endpoint provides real-time service status and statistics, including document count and database connectivity.

**Production-Ready Code**: The implementation includes comprehensive error handling, structured logging, CORS configuration, and follows best practices for API design and security.

## Project Structure

The codebase is organized as follows:

```
rag-pipeline/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and endpoints
│   └── rag_core.py          # Core RAG logic and components
├── data/
│   └── uploads/             # Temporary storage for uploaded documents
├── vectorstore/             # ChromaDB vector database directory
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Installation

### Prerequisites

Ensure you have Python 3.9 or higher installed on your system.

### Setup Steps

1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone https://github.com/LOGESH-V-N/manus-test-project.git
   cd manus-test-project
   ```

2. **Create a virtual environment** to isolate dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies** from the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables** by copying the example file and filling in your credentials:
   ```bash
   cp .env.example .env
   ```

5. **Edit the .env file** with your Mistral API key and other configuration:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   MISTRAL_API_BASE_URL=https://api.mistral.ai/v1
   EMBEDDING_MODEL=mistral.codestral-embed
   LLM_MODEL=mistral-large-latest
   CHROMA_DB_PATH=./vectorstore/chroma_db
   CHROMA_COLLECTION_NAME=documents
   API_HOST=0.0.0.0
   API_PORT=8000
   ```

## Running the Application

### Development Mode

Start the FastAPI development server with auto-reload:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### Production Mode

For production deployment, use a production-grade ASGI server:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000
```

## API Endpoints

### Health Check

**Endpoint**: `GET /health`

Returns the current service status and statistics.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database_status": "connected",
  "document_count": 42
}
```

### Document Ingestion

**Endpoint**: `POST /ingest`

Upload and ingest a document (PDF or DOCX) into the vector database.

**Request**: Multipart form data with file upload
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@document.pdf"
```

**Response**:
```json
{
  "success": true,
  "document_name": "document",
  "chunks_created": 15,
  "message": "Document 'document' ingested successfully"
}
```

### Query

**Endpoint**: `POST /query`

Submit a question to retrieve relevant information from ingested documents.

**Request**:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main topics covered?",
    "n_results": 5
  }'
```

**Response**:
```json
{
  "success": true,
  "question": "What are the main topics covered?",
  "answer": "Based on the documents, the main topics include...",
  "sources": [
    {
      "chunk": "Relevant text excerpt...",
      "relevance_score": 0.92
    }
  ]
}
```

### Statistics

**Endpoint**: `GET /stats`

Retrieve current pipeline statistics.

**Response**:
```json
{
  "success": true,
  "stats": {
    "collection_name": "documents",
    "document_count": 42,
    "db_path": "./vectorstore/chroma_db"
  }
}
```

## Configuration

All configuration is managed through environment variables defined in the `.env` file. Key variables include:

| Variable | Description | Default |
|----------|-------------|---------|
| `MISTRAL_API_KEY` | Mistral API authentication key | Required |
| `MISTRAL_API_BASE_URL` | Mistral API endpoint | https://api.mistral.ai/v1 |
| `EMBEDDING_MODEL` | Model for generating embeddings | mistral.codestral-embed |
| `LLM_MODEL` | Model for generating answers | mistral-large-latest |
| `CHROMA_DB_PATH` | Path to vector database directory | ./vectorstore/chroma_db |
| `CHUNK_SIZE` | Document chunk size in characters | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks in characters | 200 |
| `API_HOST` | Server host address | 0.0.0.0 |
| `API_PORT` | Server port number | 8000 |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `MAX_UPLOAD_SIZE_MB` | Maximum file upload size | 50 |

## Architecture

The RAG Pipeline consists of several interconnected components:

**DocumentProcessor**: Handles extraction of text from PDF and DOCX files using industry-standard libraries (PyPDF2 and python-docx).

**TextChunker**: Splits extracted text into overlapping chunks of configurable size, ensuring context preservation across chunk boundaries.

**EmbeddingGenerator**: Communicates with Mistral API to generate vector embeddings for text chunks and queries using the codestral-embed model.

**VectorStore**: Manages ChromaDB collections for storing and retrieving embeddings with cosine similarity search.

**RAGPipeline**: Orchestrates the entire workflow, coordinating document ingestion and query processing across all components.

**FastAPI Application**: Exposes REST endpoints for document ingestion, querying, health monitoring, and statistics retrieval.

## Error Handling and Logging

The pipeline implements comprehensive error handling at every stage:

- **File validation**: Checks file format and size before processing
- **API error handling**: Gracefully handles Mistral API failures with retry logic
- **Database errors**: Manages ChromaDB connection and operation failures
- **Request validation**: Validates input parameters and returns meaningful error messages

All operations are logged to both console and file (`logs/rag_pipeline.log`) with configurable log levels. Log entries include timestamps, component names, and detailed error messages for debugging.

## Performance Considerations

For optimal performance, consider the following recommendations:

**Chunk Size**: Larger chunks (1500-2000 characters) capture more context but reduce retrieval precision. Smaller chunks (500-800 characters) improve precision but may lose context.

**Embedding Batch Processing**: The system processes embeddings sequentially. For large document batches, consider implementing parallel processing with rate limiting to respect API quotas.

**Vector Database Indexing**: ChromaDB automatically creates indexes. For very large collections (>100k documents), consider implementing custom indexing strategies.

**API Rate Limiting**: Mistral API has rate limits. Implement exponential backoff and request queuing for production deployments.

## Development and Testing

To extend the pipeline with custom features:

1. Add new components in `app/rag_core.py` following the existing class structure
2. Extend FastAPI endpoints in `app/main.py` with appropriate error handling
3. Update environment variables in `.env.example` for new configuration options
4. Test thoroughly with sample documents before production deployment

## Troubleshooting

**Issue**: "Invalid API key" error
- **Solution**: Verify your Mistral API key is correctly set in the `.env` file

**Issue**: ChromaDB connection errors
- **Solution**: Ensure the `CHROMA_DB_PATH` directory exists and is writable

**Issue**: File upload failures
- **Solution**: Check file size against `MAX_UPLOAD_SIZE_MB` and verify file format is PDF or DOCX

**Issue**: Slow query responses
- **Solution**: Reduce `n_results` parameter, optimize `CHUNK_SIZE`, or upgrade to a faster server

## Security Considerations

For production deployment, implement the following security measures:

- Store API keys in secure environment variable management systems (never commit `.env` files)
- Implement authentication and authorization for API endpoints
- Use HTTPS for all API communications
- Implement rate limiting to prevent abuse
- Validate and sanitize all user inputs
- Regularly update dependencies to patch security vulnerabilities

## Contributing

To contribute improvements to the RAG Pipeline:

1. Create a feature branch for your changes
2. Implement your feature with comprehensive error handling and logging
3. Test thoroughly with various document types and queries
4. Submit a pull request with a clear description of your changes

## License

This project is provided as-is for educational and commercial use.

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository or contact the development team.

## Version History

**Version 1.0.0** (Initial Release)
- Document ingestion for PDF and DOCX files
- Embedding generation using Mistral codestral-embed
- Vector storage and retrieval with ChromaDB
- RAG-based question answering with Mistral LLM
- Health check and statistics endpoints
- Comprehensive error handling and logging
