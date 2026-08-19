import unittest
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever
from langchain_core.documents import Document

class TestRetriever(unittest.TestCase):
    def setUp(self):
        # Create test documents
        self.documents = [
            Document(page_content="This is the first test document about AI.", metadata={"source": "test1.pdf"}),
            Document(page_content="This is the second test document about Machine Learning.", metadata={"source": "test2.pdf"}),
            Document(page_content="This is the third test document about Deep Learning.", metadata={"source": "test3.pdf"})
        ]
        
        # Create embeddings
        embedder = EmbeddingGenerator()
        texts = [doc.page_content for doc in self.documents]
        embeddings = embedder.generate_embeddings(texts)
        
        # Create vector store
        self.vector_store = VectorStore()
        self.vector_store.create_index(embeddings, self.documents)
        
        # Create retriever
        self.retriever = Retriever(self.vector_store, k=2)
    
    def test_retrieval(self):
        query = "Tell me about AI"
        documents, scores = self.retriever.retrieve(query)
        
        self.assertEqual(len(documents), 2)
        self.assertEqual(len(scores), 2)
        self.assertIsInstance(documents[0], Document)

if __name__ == "__main__":
    unittest.main()