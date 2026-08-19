import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from langchain_core.documents import Document
from src.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dimension = 4
        self.vector_store = VectorStore(dimension=self.dimension)
        
        # Test documents
        self.docs = [
            Document(page_content="A", metadata={"source": "doc1.pdf"}),
            Document(page_content="B", metadata={"source": "doc1.pdf"}),
            Document(page_content="C", metadata={"source": "doc2.pdf"})
        ]
        
        self.embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype='float32')
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_create_and_search(self):
        self.vector_store.create_index(self.embeddings, self.docs)
        self.assertEqual(len(self.vector_store.documents), 3)
        
        # Search
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype='float32')
        results, scores = self.vector_store.search(query, k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "A")
        
    def test_remove_source(self):
        self.vector_store.create_index(self.embeddings, self.docs)
        
        # Remove doc1.pdf (which has 2 chunks)
        success = self.vector_store.remove_source("doc1.pdf")
        self.assertTrue(success)
        self.assertEqual(len(self.vector_store.documents), 1)
        self.assertEqual(self.vector_store.documents[0].metadata["source"], "doc2.pdf")
        
        # Test search after removal
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype='float32')
        results, scores = self.vector_store.search(query, k=1)
        # Should return C since it's the only doc left
        self.assertEqual(results[0].page_content, "C")
        
    def test_save_load(self):
        self.vector_store.create_index(self.embeddings, self.docs)
        save_path = Path(self.temp_dir) / "faiss_test"
        self.vector_store.save_index(str(save_path))
        
        new_store = VectorStore(dimension=self.dimension)
        new_store.load_index(str(save_path))
        self.assertEqual(len(new_store.documents), 3)
        self.assertEqual(new_store.documents[0].page_content, "A")

if __name__ == "__main__":
    unittest.main()
