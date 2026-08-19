from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore

class Retriever:
    def __init__(self, vector_store: VectorStore, k: int = 4):
        self.vector_store = vector_store
        self.k = k
        self.embedder = EmbeddingGenerator()
    
    def retrieve(self, query: str, k: int = None) -> Tuple[List[Document], List[float]]:
        """
        Retrieve relevant documents for a query
        """
        if k is None:
            k = self.k
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)
        
        # Search in vector store
        documents, scores = self.vector_store.search(query_embedding, k)
        
        return documents, scores
    
    def get_relevant_documents(self, query: str, k: int = None) -> List[Document]:
        """
        Get relevant documents without scores
        """
        documents, _ = self.retrieve(query, k)
        return documents
    
    def retrieve_with_context(self, query: str, k: int = None) -> Dict[str, Any]:
        """
        Retrieve documents with context and metadata
        """
        documents, scores = self.retrieve(query, k)
        
        contexts = []
        sources = []
        for doc, score in zip(documents, scores):
            contexts.append(doc.page_content)
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0),
                "score": score
            })
        
        return {
            "contexts": contexts,
            "sources": sources,
            "documents": documents,
            "scores": scores
        }