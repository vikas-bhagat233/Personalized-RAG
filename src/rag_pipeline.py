from typing import List, Tuple, Dict, Optional
from .retriever import Retriever
from .gemini_llm import GeminiLLM
import os
from pathlib import Path

class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        top_k: int = 4
    ):
        self.retriever = retriever
        self.llm = GeminiLLM(model_name=model_name, temperature=temperature)
        self.top_k = top_k
        
        # Load prompts
        self.system_prompt = self._load_prompt("system_prompt.txt")
        self.rag_prompt = self._load_prompt("rag_prompt.txt")
    
    def _load_prompt(self, filename: str) -> str:
        """
        Load prompt from file
        """
        prompt_path = Path("prompts") / filename
        if prompt_path.exists():
            with open(prompt_path, "r") as f:
                return f.read()
        return ""
    
    def query(self, question: str, k: Optional[int] = None) -> Tuple[str, List[str]]:
        """
        Process a query through the RAG pipeline
        """
        if k is None:
            k = self.top_k
        
        # Retrieve relevant documents
        retrieval_results = self.retriever.retrieve_with_context(question, k)
        contexts = retrieval_results["contexts"]
        sources = retrieval_results["sources"]
        
        # Generate response
        response = self.llm.generate_response(
            prompt=question,
            context=contexts,
            system_prompt=self.system_prompt or None
        )
        
        # Extract source citations
        source_citations = [
            f"{s['source']} (page {s['page']})" 
            for s in sources 
            if s.get('source') != 'Unknown'
        ]
        
        return response, source_citations
    
    def chat(self, message: str, history: List[Dict], k: Optional[int] = None) -> Tuple[str, List[str]]:
        """
        Chat interface with conversation history
        """
        if k is None:
            k = self.top_k
        
        # Retrieve relevant documents
        retrieval_results = self.retriever.retrieve_with_context(message, k)
        contexts = retrieval_results["contexts"]
        sources = retrieval_results["sources"]
        
        # Generate chat response
        response = self.llm.generate_chat_response(
            messages=history + [{"role": "user", "content": message}],
            context=contexts
        )
        
        # Extract source citations
        source_citations = [
            f"{s['source']} (page {s['page']})" 
            for s in sources 
            if s.get('source') != 'Unknown'
        ]
        
        return response, source_citations