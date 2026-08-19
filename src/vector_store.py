import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from langchain_core.documents import Document

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.embeddings = None
        
    def create_index(self, embeddings: np.ndarray, documents: List[Document]):
        """
        Create FAISS index from embeddings and documents
        """
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings provided")
        
        self.embeddings = embeddings
        self.documents = documents
        self.dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"[INFO] Created FAISS index with {len(documents)} documents")
    
    def search(self, query_embedding: np.ndarray, k: int = 4) -> Tuple[List[Document], List[float]]:
        """
        Search for similar documents
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index first.")
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        scores = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append(self.documents[idx])
                scores.append(float(dist))
        
        return results, scores
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Document]):
        """
        Add new documents to existing index
        """
        if self.index is None:
            self.create_index(embeddings, documents)
        else:
            self.index.add(embeddings.astype('float32'))
            self.documents.extend(documents)
            self.embeddings = np.vstack([self.embeddings, embeddings]) if self.embeddings is not None else embeddings
    
    def remove_source(self, source_name: str) -> bool:
        """
        Remove all documents and embeddings associated with a source
        """
        if not self.documents:
            return False
            
        remaining_indices = [
            i for i, doc in enumerate(self.documents)
            if doc.metadata.get("source") != source_name
        ]
        
        if len(remaining_indices) == len(self.documents):
            return False
            
        if not remaining_indices:
            self.index = None
            self.documents = []
            self.embeddings = None
            return True
            
        self.documents = [self.documents[i] for i in remaining_indices]
        if self.embeddings is not None:
            self.embeddings = self.embeddings[remaining_indices]
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self.embeddings.astype('float32'))
            
        return True

    def save_index(self, path: str):
        """
        Save FAISS index and documents
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents and metadata
        with open(path / "index.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "embeddings": self.embeddings,
                "dimension": self.dimension
            }, f)
        
        print(f"[INFO] Saved index to {path}")
    
    def load_index(self, path: str):
        """
        Load FAISS index and documents
        """
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load documents and metadata
        with open(path / "index.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.embeddings = data["embeddings"]
            self.dimension = data["dimension"]
        
        print(f"[INFO] Loaded index from {path}")
    
    def get_document(self, idx: int) -> Optional[Document]:
        """
        Get document by index
        """
        if 0 <= idx < len(self.documents):
            return self.documents[idx]
        return None