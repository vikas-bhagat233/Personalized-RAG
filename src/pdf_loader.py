import os
from pathlib import Path
from typing import List, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class PDFLoader:
    def __init__(self, pdf_paths: Union[str, List[str]] = None):
        self.documents = []
        self.pdf_paths = pdf_paths
    
    def load_documents(self, pdf_paths: Union[str, List[str]] = None) -> List[Document]:
        """
        Load PDF documents from given paths
        """
        if pdf_paths is None:
            pdf_paths = self.pdf_paths
            
        if not pdf_paths:
            return []
            
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        
        all_documents = []
        for path in pdf_paths:
            try:
                loader = PyPDFLoader(path)
                documents = loader.load()
                # Add source metadata
                for doc in documents:
                    doc.metadata["source"] = os.path.basename(path)
                all_documents.extend(documents)
                print(f"[INFO] Loaded PDF: {os.path.basename(path)}")
            except Exception as e:
                print(f"[ERROR] Error loading {path}: {str(e)}")
        
        self.documents = all_documents
        return all_documents
    
    def load_from_directory(self, directory_path: str) -> List[Document]:
        """
        Load all PDFs from a directory
        """
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        return self.load_documents([str(p) for p in pdf_files])