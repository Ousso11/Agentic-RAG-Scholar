# retriever.py
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self, embedding_model_name: str, persist_dir: str, k: int = 8, weights=(0.5, 0.5)):
        self.embeddings = OllamaEmbeddings(model=embedding_model_name)
        self.persist_dir = persist_dir
        self.k = k
        self.weights = list(weights)

    def build_hybrid_retriever(self, docs):
        try:
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
            logger.info("Vector store created.")

            bm25 = BM25Retriever.from_documents(docs)
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": self.k})

            hybrid = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=self.weights
            )
            logger.info("Hybrid retriever ready.")
            return hybrid
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise
