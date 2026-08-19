import google.generativeai as genai
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiLLM:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize Gemini LLM
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        print(f"[INFO] Loaded Gemini model: {model_name}")
    
    def generate_response(
        self,
        prompt: str,
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response using Gemini
        """
        # Prepare context
        context_text = "\n\n".join(context)
        
        # Prepare full prompt
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        
        full_prompt += f"Context:\n{context_text}\n\n"
        full_prompt += f"Question: {prompt}\n\n"
        full_prompt += "Answer:"
        
        # Generate response
        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
        )
        return response.text

    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[List[str]] = None
    ) -> str:
        """
        Generate chat response with conversation history
        """
        # Formulate history for Gemini
        gemini_history = []
        
        # The last message is the current user query, so we exclude it from history
        history_messages = messages[:-1]
        
        for msg in history_messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            gemini_history.append({
                "role": role,
                "parts": [content]
            })
            
        # Start chat with history
        chat = self.model.start_chat(history=gemini_history)
        
        # Format current prompt with context if provided
        current_query = messages[-1]["content"]
        if context:
            context_text = "\n\n".join(context)
            prompt = f"Context information:\n{context_text}\n\nQuestion: {current_query}"
        else:
            prompt = current_query
            
        # Send current message
        response = chat.send_message(prompt)
        return response.text
