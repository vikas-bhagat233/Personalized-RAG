import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pdf_loader import PDFLoader
from src.text_splitter import TextSplitter
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.gemini_llm import GeminiLLM
from src.rag_pipeline import RAGPipeline
from src.utils import Utils

# Page configuration
st.set_page_config(
    page_title="Personalized RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    with open('assets/styles.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.rag_pipeline = None
    st.session_state.chat_history = []
    st.session_state.current_docs = []

# Load CSS
if os.path.exists('assets/styles.css'):
    load_css()

def process_documents(pdf_paths):
    """Process documents and create vector store"""
    with st.spinner("Loading and processing PDFs..."):
        # Load PDFs
        loader = PDFLoader(pdf_paths)
        documents = loader.load_documents()
        
        # Split text
        splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        
        # Generate embeddings
        embedder = EmbeddingGenerator()
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedder.generate_embeddings(texts)
        
        # Create vector store
        vector_store = VectorStore()
        vector_store.create_index(embeddings, chunks)
        vector_store.save_index("database/faiss_index")
        
        return vector_store

# Title and header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if os.path.exists('assets/logo.png'):
        st.image('assets/logo.png', width=150)
    st.title("📚 Personalized RAG System")
    st.markdown("### Your Intelligent Document Assistant")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    env_api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_placeholder = "Key configured via env (hidden)" if env_api_key else "Enter your Google API Key"
    
    api_key_input = st.text_input(
        "Google API Key",
        type="password",
        value="",
        placeholder=api_key_placeholder,
        help="Enter your Google API key for Gemini to override env configuration"
    )
    
    api_key = api_key_input if api_key_input else env_api_key
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    # Model selection
    model_name = st.selectbox(
        "Select Gemini Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-3.5-flash"],
        index=0
    )
    
    # Handle dynamic updates if model or API key changes
    if 'current_model' not in st.session_state:
        st.session_state.current_model = model_name
    elif st.session_state.current_model != model_name:
        st.session_state.current_model = model_name
        if st.session_state.initialized and st.session_state.retriever:
            try:
                st.session_state.rag_pipeline = RAGPipeline(
                    retriever=st.session_state.retriever,
                    model_name=model_name
                )
            except Exception as e:
                st.error(f"Error changing model: {str(e)}")
            
    if 'current_api_key' not in st.session_state:
        st.session_state.current_api_key = api_key
    elif st.session_state.current_api_key != api_key:
        st.session_state.current_api_key = api_key
        if st.session_state.initialized and st.session_state.retriever:
            try:
                st.session_state.rag_pipeline = RAGPipeline(
                    retriever=st.session_state.retriever,
                    model_name=model_name
                )
            except Exception as e:
                st.error(f"Error updating API Key: {str(e)}")
    
    st.divider()
    
    # Upload section
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload your PDF documents for processing"
    )
    
    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files
                    upload_dir = Path("uploads/user_uploaded_files")
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    
                    pdf_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = upload_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        pdf_paths.append(str(file_path))
                    
                    # Process documents
                    st.session_state.vector_store = process_documents(pdf_paths)
                    st.session_state.retriever = Retriever(st.session_state.vector_store)
                    st.session_state.rag_pipeline = RAGPipeline(
                        retriever=st.session_state.retriever,
                        model_name=model_name
                    )
                    st.session_state.initialized = True
                    
                    st.success(f"✅ Successfully processed {len(pdf_paths)} documents!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
    
    st.divider()
    
    # Existing documents
    st.header("📁 Existing Documents")
    if st.button("📂 Load Default Documents"):
        with st.spinner("Loading default documents..."):
            try:
                pdf_dir = Path("data/pdfs")
                if pdf_dir.exists():
                    pdf_paths = [str(p) for p in pdf_dir.glob("*.pdf")]
                    if pdf_paths:
                        st.session_state.vector_store = process_documents(pdf_paths)
                        st.session_state.retriever = Retriever(st.session_state.vector_store)
                        st.session_state.rag_pipeline = RAGPipeline(
                            retriever=st.session_state.retriever,
                            model_name=model_name
                        )
                        st.session_state.initialized = True
                        st.success(f"✅ Loaded {len(pdf_paths)} documents!")
                    else:
                        st.warning("No PDF files found in data/pdfs/")
                else:
                    st.warning("Directory data/pdfs/ does not exist")
            except Exception as e:
                st.error(f"❌ Error loading documents: {str(e)}")
    
    st.divider()
    
    # Statistics
    if st.session_state.vector_store and st.session_state.vector_store.documents:
        st.header("📊 Statistics")
        unique_docs = len(set(doc.metadata.get("source") for doc in st.session_state.vector_store.documents))
        st.metric("Documents", unique_docs)
        st.metric("Total Chunks", len(st.session_state.vector_store.documents))

def main():
    # Main chat interface
    if not st.session_state.initialized:
        st.info("👈 Please upload documents or load default documents from the sidebar to get started.")
        
        # Display features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 📄 Document Upload
            Upload your PDF documents for instant processing and retrieval
            """)
        with col2:
            st.markdown("""
            ### 🧠 Smart Retrieval
            Advanced RAG system that finds the most relevant information
            """)
        with col3:
            st.markdown("""
            ### 💬 AI Chat
            Ask questions and get answers with proper context and citations
            """)
    else:
        # Document Manager Expandable Section
        with st.expander("📁 Manage Active Documents", expanded=False):
            if st.session_state.vector_store and st.session_state.vector_store.documents:
                # Group documents by source
                doc_stats = {}
                for doc in st.session_state.vector_store.documents:
                    source = doc.metadata.get("source", "Unknown")
                    if source not in doc_stats:
                        doc_stats[source] = {
                            "chunks": 0,
                            "chars": 0
                        }
                    doc_stats[source]["chunks"] += 1
                    doc_stats[source]["chars"] += len(doc.page_content)
                
                st.markdown("#### Active Documents in Database")
                
                # Create a table header
                col_name, col_chunks, col_size, col_action = st.columns([4, 2, 2, 2])
                with col_name:
                    st.markdown("**Document Name**")
                with col_chunks:
                    st.markdown("**Text Chunks**")
                with col_size:
                    st.markdown("**Est. Size**")
                with col_action:
                    st.markdown("**Action**")
                    
                st.divider()
                
                # Display each file with details and a delete button
                for source, stats in doc_stats.items():
                    col_name, col_chunks, col_size, col_action = st.columns([4, 2, 2, 2])
                    with col_name:
                        st.markdown(f"📄 `{source}`")
                    with col_chunks:
                        st.markdown(f"{stats['chunks']} chunks")
                    with col_size:
                        kb_size = round(stats['chars'] / 1024, 1)
                        st.markdown(f"{kb_size} KB")
                    with col_action:
                        if st.button("🗑️ Delete", key=f"del_{source}", type="secondary"):
                            with st.spinner(f"Removing {source}..."):
                                removed = st.session_state.vector_store.remove_source(source)
                                if removed:
                                    st.session_state.vector_store.save_index("database/faiss_index")
                                    
                                    if st.session_state.vector_store.documents:
                                        st.session_state.retriever = Retriever(st.session_state.vector_store)
                                        st.session_state.rag_pipeline = RAGPipeline(
                                            retriever=st.session_state.retriever,
                                            model_name=st.session_state.current_model
                                        )
                                    else:
                                        st.session_state.vector_store = None
                                        st.session_state.retriever = None
                                        st.session_state.rag_pipeline = None
                                        st.session_state.initialized = False
                                    
                                    st.success(f"Successfully deleted {source}!")
                                    time.sleep(1)
                                    st.rerun()
            else:
                st.info("No documents are currently indexed.")
        
        st.divider()

        # Chat interface
        st.markdown("### 💬 Chat with your Documents")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    if "sources" in message:
                        with st.expander("📚 Sources"):
                            for source in message["sources"]:
                                st.write(f"- {source}")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Display user message
            st.chat_message("user").write(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get response from RAG pipeline with chat history
                        response, sources = st.session_state.rag_pipeline.chat(
                            message=prompt,
                            history=st.session_state.chat_history
                        )
                        
                        # Display response
                        st.write(response)
                        if sources:
                            with st.expander("📚 Sources"):
                                for source in sources:
                                    st.write(f"- {source}")
                        
                        # Save to history
                        st.session_state.chat_history.append({"role": "user", "content": prompt})
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()