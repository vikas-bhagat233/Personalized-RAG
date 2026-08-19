# Personalized RAG System 📚

A smart Retrieval-Augmented Generation (RAG) application that allows users to upload personal documents, extract useful information, and ask questions in natural language. The system uses Google Gemini for answer generation and FAISS for fast semantic retrieval, making it useful for document-based Q&A, knowledge lookup, and personal knowledge assistants.

## Overview

This project is designed to help users interact with their own PDFs and documents without manually reading every page. Instead of scanning files manually, the application processes the uploaded documents, chunks them into meaningful segments, converts them into embeddings, stores them in a vector database, and then retrieves the most relevant content before generating a final response.

It is especially useful for:
- personal document search
- research and study notes
- PDF-based knowledge assistants
- internal knowledge retrieval systems
- document Q&A workflows

## Key Features

- PDF upload and processing
- Text chunking and intelligent segmentation
- Embedding generation for semantic search
- FAISS-based vector indexing for fast retrieval
- Question answering using Gemini AI
- Source-aware responses with relevant document references
- Streamlit-based interactive chat interface
- Persistent storage for processed knowledge

## How It Works

1. The user uploads one or more PDF files.
2. The system extracts text content from the documents.
3. The text is split into smaller chunks for better retrieval efficiency.
4. Each chunk is converted into embeddings using a sentence-transformer model.
5. These embeddings are stored in a FAISS vector database.
6. When the user asks a question, the system retrieves the most relevant chunks.
7. Google Gemini generates a concise and context-aware answer using the retrieved content.

## Tools & Technologies

Python, Streamlit, LangChain, LangChain Community, Google Generative AI, Gemini API, FAISS, PyPDF, Sentence Transformers, NumPy, Pillow, dotenv, GitHub

## Project Structure

```bash
Personalized-RAG/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── assets/
│   └── styles.css
├── data/
│   └── pdfs/
├── database/
│   └── faiss_index/
├── prompts/
│   ├── rag_prompt.txt
│   └── system_prompt.txt
├── src/
│   ├── embeddings.py
│   ├── gemini_llm.py
│   ├── pdf_loader.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── text_splitter.py
│   ├── utils.py
│   └── vector_store.py
├── tests/
│   ├── test_embeddings.py
│   ├── test_pdf_loader.py
│   ├── test_pipeline.py
│   ├── test_retriever.py
│   └── test_vector_store.py
└── uploads/
    └── user_uploaded_files/
```

## Prerequisites

- Python 3.10+
- Google API key for Gemini
- Internet access for model and package downloads

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Personalized-RAG.git
cd Personalized-RAG
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add your API key:

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Run the Application

```bash
streamlit run app.py
```

The app will open in your browser and allow you to upload PDFs and start asking questions.

## Use Cases

- Ask questions about uploaded PDFs
- Retrieve relevant sections from long documents
- Build a private document assistant for personal data
- Search stored knowledge with semantic similarity

## Deployment

This project can be deployed on GitHub and hosted using platforms such as Streamlit Community Cloud, Hugging Face Spaces, or a custom Python hosting service. For public deployment, store the Google API key as a secret environment variable.

## License

This project is intended for educational and personal use. You may modify and extend it as needed for your own projects.

## Notes

The system is best suited for personal and private document knowledge retrieval. Its performance depends on the quality of the uploaded PDFs, embedding model, and the chosen Gemini model parameters.
