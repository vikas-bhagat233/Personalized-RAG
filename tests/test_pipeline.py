import unittest
from unittest.mock import Mock, patch
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_store import VectorStore

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        # Mock retriever
        self.mock_retriever = Mock(spec=Retriever)
        
        # Create pipeline
        self.pipeline = RAGPipeline(
            retriever=self.mock_retriever,
            model_name="gemini-2.5-flash"
        )
    
    @patch('src.gemini_llm.GeminiLLM.generate_response')
    def test_query(self, mock_generate):
        # Mock retriever response
        mock_retriever_response = {
            "contexts": ["Test context 1", "Test context 2"],
            "sources": [{"source": "test.pdf", "page": 1, "score": 0.5}, {"source": "test.pdf", "page": 2, "score": 0.3}],
            "documents": [],
            "scores": [0.5, 0.3]
        }
        self.mock_retriever.retrieve_with_context.return_value = mock_retriever_response
        
        # Mock LLM response
        mock_generate.return_value = "Test response"
        
        # Test query
        response, sources = self.pipeline.query("Test question")
        
        self.assertEqual(response, "Test response")
        self.assertEqual(len(sources), 2)

if __name__ == "__main__":
    unittest.main()