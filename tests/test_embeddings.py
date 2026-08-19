import unittest
from src.embeddings import EmbeddingGenerator
import numpy as np

class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.embedder = EmbeddingGenerator()
    
    def test_embedding_generation(self):
        texts = ["Hello world", "This is a test"]
        embeddings = self.embedder.generate_embeddings(texts)
        
        self.assertEqual(len(embeddings), 2)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[1], 384)  # all-MiniLM-L6-v2 dimension
    
    def test_single_embedding(self):
        text = "Hello world"
        embedding = self.embedder.generate_embedding(text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[0], 384)

if __name__ == "__main__":
    unittest.main()