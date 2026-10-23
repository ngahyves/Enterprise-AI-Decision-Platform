# src/retrieval/vector_store.py
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


class VectorStoreIndexer:
    """Handles native hybrid indexing (Dense + Sparse) of JSONL chunks into Qdrant."""

    def __init__(self):
        self.collection_name = env_settings.qdrant_collection_name
        self.host = env_settings.qdrant_host
        self.port = env_settings.qdrant_port
        self.chunks_dir = Path(config["paths"]["data_processed_chunks"])

        # 1. Initializing Qdrant client
        self.client = QdrantClient(host=self.host, port=self.port)

        # 2. Initializing dense embedding model (Hugging Face)
        embedding_model_name = config["retrieval"]["embedding"]["model_name"]
        logger.info(f"Loading dense embedding model: {embedding_model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 2},
        )

        # 3. Initializing sparse "Qdrant/bm25" embedding model (FastEmbed BM25 native)
        logger.info("Loading native sparse model (FastEmbed Qdrant/bm25)...")
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    def load_jsonl_chunks(self, jsonl_path: Path) -> List[Document]:
        """Load chunks from a JSONL file as LangChain Documents.

        The ingestion pipeline already wrote the final text (with bank/year/
        section prefix) directly into `page_content`. No post-processing here.
        """
        documents = []
        if not jsonl_path.exists():
            logger.error(f"File not found: {jsonl_path}")
            return documents

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                documents.append(
                    Document(
                        page_content=data["page_content"],
                        metadata=data["metadata"],
                    )
                )

        logger.info(f"Loaded {len(documents)} documents from {jsonl_path.name}")
        return documents

    def index_all_chunks(self, force_recreate: bool = True):
        """
        Index all JSONL files from the chunks directory into Qdrant with
        native Hybrid support (Dense + Sparse) using batching.
        """
        jsonl_files = list(self.chunks_dir.glob("*.jsonl"))

        if not jsonl_files:
            logger.warning(f"No .jsonl files found in {self.chunks_dir}")
            return

        all_documents: List[Document] = []
        for file_path in jsonl_files:
            docs = self.load_jsonl_chunks(file_path)
            all_documents.extend(docs)

        if not all_documents:
            logger.warning("No documents to index.")
            return

        # Delete existing collection if force_recreate
        if force_recreate and self.client.collection_exists(self.collection_name):
            logger.info(
                f"Deleting existing collection '{self.collection_name}' "
                f"to recreate with Hybrid schema..."
            )
            self.client.delete_collection(self.collection_name)

        # Create collection with dense + sparse vectors
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Creating native hybrid collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                ),
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier="idf"),
                },
            )

        logger.info(
            f"Starting batch indexing of {len(all_documents)} documents "
            f"into Qdrant Native Hybrid..."
        )

        # Split into batches of 10 documents to avoid memory/network timeouts
        batch_size = 10
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} documents)...")

            try:
                QdrantVectorStore.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    sparse_embedding=self.sparse_embeddings,
                    sparse_vector_name="sparse",
                    url=f"http://{self.host}:{self.port}",
                    collection_name=self.collection_name,
                    retrieval_mode=RetrievalMode.HYBRID,
                    force_recreate=False,
                )
            except Exception as e:
                logger.error(f"Error at batch {i // batch_size + 1}: {str(e)}")
                raise e

        logger.info("Successfully indexed all reports into Qdrant with Hybrid Native structure!")


if __name__ == "__main__":
    indexer = VectorStoreIndexer()
    indexer.index_all_chunks(force_recreate=True)
