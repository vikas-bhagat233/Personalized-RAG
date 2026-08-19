import unittest
from unittest.mock import patch, Mock
from src.pdf_loader import PDFLoader
from langchain_core.documents import Document

class TestPDFLoader(unittest.TestCase):
    @patch('src.pdf_loader.PyPDFLoader')
    def test_load_documents_single(self, mock_pypdf_loader):
        # Setup mock loader to return mock documents
        mock_instance = mock_pypdf_loader.return_value
        mock_instance.load.return_value = [
            Document(page_content="Page 1 content", metadata={"page": 0})
        ]
        
        # Test constructor taking path
        loader = PDFLoader("dummy.pdf")
        self.assertEqual(loader.pdf_paths, "dummy.pdf")
        
        # Test loading documents
        docs = loader.load_documents()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Page 1 content")
        self.assertEqual(docs[0].metadata["source"], "dummy.pdf")
        
    @patch('src.pdf_loader.PyPDFLoader')
    def test_load_documents_list(self, mock_pypdf_loader):
        mock_instance = mock_pypdf_loader.return_value
        mock_instance.load.side_effect = lambda: [
            Document(page_content="Doc content", metadata={"page": 0})
        ]
        
        # Test calling load_documents with list directly
        loader = PDFLoader()
        docs = loader.load_documents(["doc1.pdf", "doc2.pdf"])
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["source"], "doc1.pdf")
        self.assertEqual(docs[1].metadata["source"], "doc2.pdf")

if __name__ == "__main__":
    unittest.main()
