from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional

class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        if separators is None:
            separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        """
        return self.splitter.split_documents(documents)
    
    def split_text(self, text: str) -> List[str]:
        """
        Split a single text into chunks
        """
        return self.splitter.split_text(text)