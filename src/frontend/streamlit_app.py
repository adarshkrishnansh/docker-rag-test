import streamlit as st
import httpx
from typing import Optional, List
import time

# Configure the page
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://rag-app:8000"

class RAGClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
    
    def health_check(self) -> Optional[dict]:
        try:
            response = self.client.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API connection failed: {e}")
            return None
    
    def upload_files(self, files: List) -> Optional[dict]:
        try:
            files_data = []
            for uploaded_file in files:
                files_data.append(("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))
            
            response = self.client.post(f"{self.base_url}/upload", files=files_data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Upload failed: {e}")
            return None
    
    def query_documents(self, query: str, k: int = 5) -> Optional[dict]:
        try:
            payload = {"query": query, "k": k}
            response = self.client.post(f"{self.base_url}/query", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Query failed: {e}")
            return None
    
    def chat(self, message: str, k: int = 5, use_context: bool = True, temperature: float = 0.7) -> Optional[dict]:
        try:
            payload = {
                "message": message,
                "k": k,
                "use_context": use_context,
                "temperature": temperature
            }
            response = self.client.post(f"{self.base_url}/chat", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Chat failed: {e}")
            return None
    
    def get_document_count(self) -> Optional[dict]:
        try:
            response = self.client.get(f"{self.base_url}/documents/count")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get document count: {e}")
            return None
    
    def clear_documents(self) -> Optional[dict]:
        try:
            response = self.client.delete(f"{self.base_url}/documents")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to clear documents: {e}")
            return None

# Initialize client
@st.cache_resource
def get_rag_client():
    return RAGClient(API_BASE_URL)

def main():
    st.title("📚 RAG Document Assistant")
    st.markdown("Upload documents and ask questions using Retrieval-Augmented Generation")
    
    # Initialize client
    client = get_rag_client()
    
    # Sidebar for system info and controls
    with st.sidebar:
        st.header("System Status")
        
        # Health check
        health = client.health_check()
        if health:
            st.success("✅ API Connected")
            st.info(f"Version: {health.get('version', 'Unknown')}")
            st.info(f"Documents: {health.get('document_count', 0)}")
        else:
            st.error("❌ API Disconnected")
            st.stop()
        
        st.divider()
        
        # Document management
        st.header("Document Management")
        
        # Get current document count
        doc_count = client.get_document_count()
        if doc_count:
            st.metric("Documents in Database", doc_count.get('count', 0))
        
        # Clear documents button
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                result = client.clear_documents()
                if result:
                    st.success("Documents cleared successfully!")
                    st.rerun()
                st.session_state.confirm_clear = False
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm deletion")
        
        st.divider()
        
        # Settings
        st.header("Chat Settings")
        context_enabled = st.checkbox("Use document context", value=True)
        temperature = st.slider("Response creativity", 0.0, 2.0, 0.7, 0.1)
        max_results = st.slider("Max search results", 1, 10, 5)
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Search", "📤 Upload"])
    
    with tab1:
        st.header("Chat with Your Documents")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["message"])
                
                with st.chat_message("assistant"):
                    st.write(chat["response"])
                    if chat.get("sources"):
                        with st.expander("📎 Sources"):
                            for i, source in enumerate(chat["sources"], 1):
                                st.write(f"{i}. {source}")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = client.chat(
                        message=prompt,
                        k=max_results,
                        use_context=context_enabled,
                        temperature=temperature
                    )
                
                if response:
                    st.write(response["response"])
                    
                    # Show sources if available
                    if response.get("sources"):
                        with st.expander("📎 Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.write(f"{i}. {source}")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "message": prompt,
                        "response": response["response"],
                        "sources": response.get("sources", [])
                    })
                else:
                    st.error("Failed to get response. Please try again.")
    
    with tab2:
        st.header("Search Documents")
        
        # Search form
        with st.form("search_form"):
            search_query = st.text_input("Enter your search query:", placeholder="e.g., What is the main topic?")
            search_k = st.slider("Number of results", 1, 10, 5, key="search_k")
            search_submitted = st.form_submit_button("🔍 Search")
        
        if search_submitted and search_query:
            with st.spinner("Searching..."):
                results = client.query_documents(search_query, k=search_k)
            
            if results and results.get("results"):
                st.success(f"Found {results['total_results']} results")
                
                for i, result in enumerate(results["results"], 1):
                    with st.expander(f"Result {i} - {result['source']} (Score: {result['score']:.3f})"):
                        st.write(result["content"])
                        
                        # Show metadata if available
                        if result.get("metadata"):
                            st.json(result["metadata"])
            else:
                st.warning("No results found for your query.")
    
    with tab3:
        st.header("Upload Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["pdf", "txt", "md", "docx", "html"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, MD, DOCX, HTML"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size} bytes)")
            
            if st.button("📤 Upload and Process", type="primary"):
                with st.spinner("Uploading and processing documents..."):
                    result = client.upload_files(uploaded_files)
                
                if result:
                    st.success("Upload successful!")
                    st.info(f"Processed {result['documents_processed']} documents")
                    st.info(f"Created {result['chunks_created']} chunks")
                    
                    if result.get("errors"):
                        st.warning("Some errors occurred:")
                        for error in result["errors"]:
                            st.error(error)
                    
                    # Refresh the page to update document count
                    time.sleep(1)
                    st.rerun()
        
        # Instructions
        st.info("""
        **How to use:**
        1. Select one or more documents to upload
        2. Click 'Upload and Process' to add them to the knowledge base
        3. Use the Chat or Search tabs to interact with your documents
        
        **Supported formats:** PDF, TXT, MD, DOCX, HTML
        """)

if __name__ == "__main__":
    main()