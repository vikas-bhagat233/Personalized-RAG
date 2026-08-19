import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        You can also use Google's embedding models if preferred
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        """
        return self.model.encode([text], convert_to_numpy=True)[0]