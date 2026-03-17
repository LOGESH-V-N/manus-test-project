"""
Core RAG (Retrieval-Augmented Generation) logic module.

This module handles document processing, embedding generation, vector storage,
and retrieval-augmented generation using Mistral API.
"""

import logging
import os
from typing import Optional
import PyPDF2
from docx import Document
import chromadb
from chromadb.config import Settings
import requests

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading and text extraction from PDF and DOCX files."""

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text from the PDF.

        Raises:
            Exception: If PDF processing fails.
        """
        try:
            text = ""
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            logger.info(f"Successfully extracted text from PDF: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """
        Extract text from a DOCX file.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            Extracted text from the DOCX file.

        Raises:
            Exception: If DOCX processing fails.
        """
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logger.info(f"Successfully extracted text from DOCX: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            raise

    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract text from a document based on file extension.

        Args:
            file_path: Path to the document file.

        Returns:
            Extracted text from the document.

        Raises:
            ValueError: If file format is not supported.
            Exception: If text extraction fails.
        """
        if file_path.lower().endswith(".pdf"):
            return DocumentProcessor.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(".docx"):
            return DocumentProcessor.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")


class TextChunker:
    """Handles text chunking with configurable chunk size and overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Size of each chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks


class EmbeddingGenerator:
    """Generates embeddings using Mistral API."""

    def __init__(self, api_key: str, model: str = "mistral.codestral-embed"):
        """
        Initialize the embedding generator.

        Args:
            api_key: Mistral API key.
            model: Embedding model name.
        """
        self.api_key = api_key
        self.model = model
        self.api_base_url = os.getenv("MISTRAL_API_BASE_URL", "https://api.mistral.ai/v1")

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a text using Mistral API.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            Exception: If embedding generation fails.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "input": text,
            }
            response = requests.post(
                f"{self.api_base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            logger.debug(f"Generated embedding with dimension {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                logger.info(f"Generated embedding {i + 1}/{len(texts)}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for text {i}: {str(e)}")
                raise
        return embeddings


class VectorStore:
    """Manages ChromaDB vector storage for document embeddings."""

    def __init__(self, db_path: str, collection_name: str = "documents"):
        """
        Initialize the vector store.

        Args:
            db_path: Path to the ChromaDB directory.
            collection_name: Name of the collection to use.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        os.makedirs(db_path, exist_ok=True)

        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Initialized ChromaDB collection: {collection_name}")

    def add_documents(
        self, texts: list[str], embeddings: list[list[float]], document_name: str
    ) -> None:
        """
        Add documents with embeddings to the vector store.

        Args:
            texts: List of text chunks.
            embeddings: List of embedding vectors.
            document_name: Name of the document being added.

        Raises:
            Exception: If adding documents fails.
        """
        try:
            ids = [f"{document_name}_{i}" for i in range(len(texts))]
            metadatas = [{"document": document_name, "chunk_index": i} for i in range(len(texts))]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"Added {len(texts)} chunks to vector store for document: {document_name}")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def search(self, query_embedding: list[float], n_results: int = 5) -> dict:
        """
        Search for similar documents using embedding.

        Args:
            query_embedding: Embedding vector of the query.
            n_results: Number of results to return.

        Returns:
            Dictionary containing search results with documents and distances.

        Raises:
            Exception: If search fails.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )
            logger.info(f"Retrieved {len(results['documents'][0])} relevant chunks")
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "db_path": self.db_path,
        }


class RAGPipeline:
    """Main RAG pipeline orchestrating document ingestion and query answering."""

    def __init__(
        self,
        mistral_api_key: str,
        chroma_db_path: str,
        embedding_model: str = "mistral.codestral-embed",
        llm_model: str = "mistral-large-latest",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            mistral_api_key: Mistral API key.
            chroma_db_path: Path to ChromaDB directory.
            embedding_model: Embedding model name.
            llm_model: LLM model name.
            chunk_size: Document chunk size.
            chunk_overlap: Chunk overlap size.
        """
        self.embedding_generator = EmbeddingGenerator(mistral_api_key, embedding_model)
        self.vector_store = VectorStore(chroma_db_path)
        self.text_chunker = TextChunker(chunk_size, chunk_overlap)
        self.llm_model = llm_model
        self.mistral_api_key = mistral_api_key
        self.api_base_url = os.getenv("MISTRAL_API_BASE_URL", "https://api.mistral.ai/v1")
        logger.info("RAG Pipeline initialized successfully")

    def ingest_document(self, file_path: str, document_name: str) -> dict:
        """
        Ingest a document into the RAG pipeline.

        Args:
            file_path: Path to the document file.
            document_name: Name to identify the document.

        Returns:
            Dictionary with ingestion results.

        Raises:
            Exception: If ingestion fails.
        """
        try:
            logger.info(f"Starting document ingestion: {document_name}")

            # Extract text
            text = DocumentProcessor.extract_text(file_path)
            logger.info(f"Extracted {len(text)} characters from document")

            # Chunk text
            chunks = self.text_chunker.chunk_text(text)
            logger.info(f"Split document into {len(chunks)} chunks")

            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings_batch(chunks)
            logger.info(f"Generated embeddings for {len(embeddings)} chunks")

            # Store in vector database
            self.vector_store.add_documents(chunks, embeddings, document_name)

            logger.info(f"Successfully ingested document: {document_name}")
            return {
                "success": True,
                "document_name": document_name,
                "chunks_created": len(chunks),
                "message": f"Document '{document_name}' ingested successfully",
            }
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to ingest document: {str(e)}",
            }

    def query(self, question: str, n_results: int = 5) -> dict:
        """
        Query the RAG pipeline and generate an answer.

        Args:
            question: User question.
            n_results: Number of relevant chunks to retrieve.

        Returns:
            Dictionary with answer and source chunks.

        Raises:
            Exception: If query processing fails.
        """
        try:
            logger.info(f"Processing query: {question}")

            # Generate embedding for the question
            question_embedding = self.embedding_generator.generate_embedding(question)
            logger.info("Generated embedding for question")

            # Search for relevant chunks
            search_results = self.vector_store.search(question_embedding, n_results)
            relevant_chunks = search_results["documents"][0]
            distances = search_results["distances"][0]

            if not relevant_chunks:
                logger.warning("No relevant chunks found for query")
                return {
                    "success": True,
                    "question": question,
                    "answer": "No relevant documents found to answer this question.",
                    "sources": [],
                }

            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")

            # Generate answer using Mistral LLM
            context = "\n\n".join(relevant_chunks)
            answer = self._generate_answer(question, context)

            logger.info("Generated answer successfully")
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "chunk": chunk,
                        "relevance_score": 1 - distance,
                    }
                    for chunk, distance in zip(relevant_chunks, distances)
                ],
            }
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "success": False,
                "question": question,
                "error": str(e),
            }

    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer using Mistral LLM.

        Args:
            question: User question.
            context: Relevant context from vector store.

        Returns:
            Generated answer.

        Raises:
            Exception: If LLM call fails.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so.",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
            }
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
            logger.debug(f"Generated answer with {len(answer)} characters")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
