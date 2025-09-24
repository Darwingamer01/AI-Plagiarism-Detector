"""
Sprint 2: Enhanced Streamlit Frontend with Dual Mode Support
- Local Mode: Uses embedded DocumentProcessor (original functionality)
- API Mode: Calls FastAPI backend when BACKEND_URL is set
"""
import streamlit as st
import requests
import numpy as np
import json
import os
import uuid
import mimetypes
import tempfile
from typing import List, Dict, Any
from datetime import datetime
import re

# API Configuration - NEW SPRINT 2 ADDITION
BACKEND_URL = os.getenv("BACKEND_URL")
API_KEY = os.getenv("API_KEY", "demo-secret")

# Try FAISS, fallback to scikit-learn if FAISS fails on Windows
FAISS_AVAILABLE = True
try:
    import faiss
except ImportError:
    FAISS_AVAILABLE = False
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document

# Configuration
SIMILARITY_THRESHOLD = 0.88
CHUNK_SIZE = 300
OVERLAP = 50
DATA_DIR = "./data"
INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
MAX_FILE_SIZE = 5_000_000  # 5MB limit

# NEW: API Client Class for FastAPI Backend Communication
class APIClient:
    """Client for communicating with FastAPI backend"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = 30
    
    def health_check(self):
        """Check if backend is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_status(self):
        """Get system status from backend"""
        try:
            response = requests.get(
                f"{self.base_url}/status",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Backend connection failed: {e}")
            return None
    
    def ingest_documents(self, files):
        """Ingest documents via API"""
        try:
            files_data = []
            for uploaded_file in files:
                files_data.append(
                    ('files', (uploaded_file.name, uploaded_file.getvalue(), 'text/plain'))
                )
            
            response = requests.post(
                f"{self.base_url}/ingest",
                headers=self.headers,
                files=files_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Ingestion failed: {e}")
            return None
    
    def check_similarity(self, uploaded_file):
        """Check similarity via API"""
        try:
            files_data = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/plain')
            }
            
            response = requests.post(
                f"{self.base_url}/check",
                headers=self.headers,
                files=files_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Similarity check failed: {e}")
            return None

# Existing Classes (Preserved from your original file)
class IndexWrapper:
    """Wrapper that uses FAISS if available, otherwise scikit-learn"""
    def __init__(self, dim):
        self.dim = dim
        self.ntotal = 0
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = NearestNeighbors(n_neighbors=10, metric='cosine')
            self._vectors = []
            self._fitted = False

    def add(self, vectors):
        if FAISS_AVAILABLE:
            self.index.add(vectors)
            self.ntotal = self.index.ntotal
        else:
            if len(self._vectors) == 0:
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])
            self.ntotal = len(self._vectors)
            self.index.fit(self._vectors)
            self._fitted = True

    def search(self, query_vector, k=5):
        if FAISS_AVAILABLE:
            return self.index.search(query_vector, k)
        else:
            if not self._fitted or len(self._vectors) == 0:
                return np.array([[0.0] * k]), np.array([[-1] * k])
            
            # Ensure k doesn't exceed available vectors
            k = min(k, len(self._vectors))
            distances, indices = self.index.kneighbors(query_vector, n_neighbors=k)
            # Convert distances to similarity scores (1 - distance for cosine)
            scores = 1 - distances
            return scores, indices

def _get_file_ext(uploaded_file):
    """Robust file extension detection with MIME type fallback"""
    name = uploaded_file.name.lower()
    if name.endswith('.pdf'):
        return 'pdf'
    if name.endswith('.docx') or name.endswith('.doc'):
        return 'docx'
    if name.endswith('.txt'):
        return 'txt'
    
    # Fallback to MIME type
    mime, _ = mimetypes.guess_type(name)
    if mime:
        if 'pdf' in mime:
            return 'pdf'
        elif 'word' in mime or 'document' in mime:
            return 'docx'
        elif 'text' in mime:
            return 'txt'
    
    # Final fallback - try uploaded_file.type if available
    try:
        if uploaded_file.type:
            if 'pdf' in uploaded_file.type:
                return 'pdf'
            elif 'word' in uploaded_file.type:
                return 'docx'
            elif 'text' in uploaded_file.type:
                return 'txt'
    except:
        pass
    
    return None

class DocumentProcessor:
    def __init__(self):
        self.model = None
        self.index = None
        self.metadata = {}
        self._model_loaded = False

    def load_model(self):
        """Load the sentence transformer model lazily"""
        if not self._model_loaded:
            try:
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self._model_loaded = True
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.error("Try: pip install torch sentence-transformers")
                st.stop()

    def extract_text(self, uploaded_file) -> str:
        """Extract text from uploaded file with robust file type detection"""
        # Check file size
        try:
            file_size = len(uploaded_file.getvalue())
            if file_size > MAX_FILE_SIZE:
                st.error(f"File too large ({file_size/1024/1024:.1f}MB). Max size: {MAX_FILE_SIZE/1024/1024:.1f}MB")
                return ""
        except:
            pass  # If getvalue() fails, proceed anyway

        file_type = _get_file_ext(uploaded_file)
        
        if file_type == 'txt':
            try:
                return str(uploaded_file.read(), "utf-8", errors='ignore')
            except Exception as e:
                st.error(f"Error reading text file: {e}")
                return ""
                
        elif file_type == 'pdf':
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file.flush()
                    
                    text = ""
                    with pdfplumber.open(tmp_file.name) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + " "
                    
                    os.unlink(tmp_file.name)
                    return text.strip()
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                return ""
                
        elif file_type == 'docx':
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file.flush()
                    
                    doc = Document(tmp_file.name)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    
                    os.unlink(tmp_file.name)
                    return text
            except Exception as e:
                st.error(f"Error reading DOCX: {e}")
                return ""
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return ""

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Improved text chunking with proper character position tracking"""
        # Better sentence splitting that preserves boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        char_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Approximate token count (words * 1.3)
            current_words = len(current_chunk.split())
            sentence_words = len(sentence.split())
            
            if current_words + sentence_words > CHUNK_SIZE and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_pos': char_pos - len(current_chunk),
                    'end_pos': char_pos
                })
                
                # Create overlap
                words = current_chunk.split()
                overlap_words = words[-OVERLAP:] if len(words) > OVERLAP else words
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence
            
            char_pos += len(sentence) + 1
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'start_pos': char_pos - len(current_chunk),
                'end_pos': char_pos
            })
        
        return chunks

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings with safe normalization"""
        if not self._model_loaded:
            self.load_model()
            
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Safe normalization - protect against zero norms
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9  # Prevent division by zero
        embeddings = embeddings / norms
        
        return embeddings.astype('float32')

    def load_index(self):
        """Load existing index and metadata with validation"""
        os.makedirs(DATA_DIR, exist_ok=True)
        
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            try:
                if FAISS_AVAILABLE:
                    faiss_index = faiss.read_index(INDEX_FILE)
                    self.index = IndexWrapper(384)
                    self.index.index = faiss_index
                    self.index.ntotal = faiss_index.ntotal
                else:
                    # Load sklearn fallback if vectors exist
                    numpy_file = INDEX_FILE + ".npy"
                    if os.path.exists(numpy_file):
                        try:
                            vectors = np.load(numpy_file)
                            self.index = IndexWrapper(384)
                            self.index._vectors = vectors
                            self.index.ntotal = vectors.shape[0]
                            self.index._fitted = True
                            # Sync metadata counter
                            self.metadata["_next_index"] = vectors.shape[0]
                        except Exception as e:
                            st.warning(f"Could not load sklearn vectors: {e}")
                            self.create_new_index()
                            return
                    else:
                        # No existing index, create new
                        self.create_new_index()
                        return

                with open(METADATA_FILE, 'r') as f:
                    self.metadata = json.load(f)

                # Validate consistency
                if self.index.ntotal != len([k for k in self.metadata.keys() if k.isdigit()]):
                    st.warning("⚠️ Index/metadata size mismatch. Rebuilding index for safety.")
                    self.create_new_index()
                    return

                total_docs = len(set(chunk.get('doc_id', '') for chunk in self.metadata.values() if isinstance(chunk, dict)))
                st.success(f"✅ Loaded index: {self.index.ntotal} chunks from {total_docs} documents")

            except Exception as e:
                st.warning(f"Could not load existing index: {e}")
                self.create_new_index()
        else:
            self.create_new_index()

    def create_new_index(self):
        """Create new index"""
        dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = IndexWrapper(dimension)
        self.metadata = {"_next_doc_id": 0, "_next_index": 0}

    def save_index(self):
        """Save index and metadata"""
        os.makedirs(DATA_DIR, exist_ok=True)
        
        if FAISS_AVAILABLE and hasattr(self.index, 'index') and hasattr(self.index.index, 'ntotal'):
            faiss.write_index(self.index.index, INDEX_FILE)
        elif not FAISS_AVAILABLE and hasattr(self.index, '_vectors') and len(self.index._vectors) > 0:
            # Save numpy vectors for sklearn fallback
            np.save(INDEX_FILE + ".npy", self.index._vectors)
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def ingest_document(self, uploaded_file) -> Dict[str, Any]:
        """Ingest document with robust error handling"""
        try:
            text = self.extract_text(uploaded_file)
            if not text or len(text.strip()) < 10:
                return {"success": False, "error": "Could not extract meaningful text"}

            chunks = self.chunk_text(text)
            if not chunks:
                return {"success": False, "error": "No text chunks created"}

            # Generate safe, unique doc ID
            next_id = self.metadata.get("_next_doc_id", 0)
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{next_id}"
            self.metadata["_next_doc_id"] = next_id + 1

            chunk_texts = [chunk['text'] for chunk in chunks]
            
            # Create embeddings
            embeddings = self.create_embeddings(chunk_texts)

            # Add to index - USE METADATA COUNTER
            start_idx = int(self.metadata.get("_next_index", 0))
            self.index.add(embeddings)

            # Update metadata
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                self.metadata[str(start_idx + i)] = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "filename": uploaded_file.name,
                    "text": chunk['text'],
                    "start_pos": chunk.get('start_pos', 0),
                    "end_pos": chunk.get('end_pos', 0)
                }

            # Update next index counter
            self.metadata["_next_index"] = start_idx + len(chunks)

            self.save_index()
            
            return {
                "success": True,
                "doc_id": doc_id,
                "chunks_added": len(chunks),
                "filename": uploaded_file.name
            }
            
        except Exception as e:
            return {"success": False, "error": f"Processing failed: {str(e)}"}

    def check_similarity(self, uploaded_file, top_k: int = 5) -> Dict[str, Any]:
        """Check similarity with enhanced error handling"""
        try:
            if self.index.ntotal == 0:
                return {"error": "No documents in index. Please ingest documents first."}

            text = self.extract_text(uploaded_file)
            if not text:
                return {"error": "Could not extract text from query document"}

            chunks = self.chunk_text(text)
            if not chunks:
                return {"error": "Could not create text chunks from query document"}

            chunk_texts = [chunk['text'] for chunk in chunks]
            query_embeddings = self.create_embeddings(chunk_texts)

            # Get dynamic thresholds
            dynamic_similarity_threshold = st.session_state.get('similarity_threshold', SIMILARITY_THRESHOLD)

            # Search for each query chunk
            all_matches = []
            high_similarity_count = 0

            for i, query_embedding in enumerate(query_embeddings):
                scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and str(idx) in self.metadata:
                        metadata = self.metadata[str(idx)]
                        if isinstance(metadata, dict) and 'doc_id' in metadata:
                            # Use dynamic threshold
                            if score >= dynamic_similarity_threshold:
                                high_similarity_count += 1

                            all_matches.append({
                                "chunk_id": metadata.get("chunk_id", "unknown"),
                                "doc_id": metadata["doc_id"],
                                "filename": metadata.get("filename", "unknown"),
                                "score": float(score),
                                "matched_text": metadata.get("text", "")[:200] + "..." if len(metadata.get("text", "")) > 200 else metadata.get("text", ""),
                                "query_text": chunks[i]['text'][:200] + "..." if len(chunks[i]['text']) > 200 else chunks[i]['text']
                            })

            # Aggregate by document
            doc_matches = {}
            for match in all_matches:
                doc_id = match["doc_id"]
                if doc_id not in doc_matches:
                    doc_matches[doc_id] = {
                        "doc_id": doc_id,
                        "filename": match["filename"],
                        "matches": [],
                        "max_score": 0,
                        "high_similarity_matches": 0
                    }

                doc_matches[doc_id]["matches"].append(match)
                doc_matches[doc_id]["max_score"] = max(doc_matches[doc_id]["max_score"], match["score"])
                # Use dynamic threshold for counting
                if match["score"] >= dynamic_similarity_threshold:
                    doc_matches[doc_id]["high_similarity_matches"] += 1

            # Sort by similarity
            sorted_matches = sorted(doc_matches.values(), key=lambda x: x["max_score"], reverse=True)

            # Calculate final scores with dynamic plagiarism threshold
            total_chunks = len(chunks)
            aggregated_score = high_similarity_count / total_chunks if total_chunks > 0 else 0
            dynamic_plagiarism_threshold = st.session_state.get('plagiarism_threshold', 0.3)
            plagiarism_flag = aggregated_score > dynamic_plagiarism_threshold

            return {
                "query_filename": uploaded_file.name,
                "total_query_chunks": total_chunks,
                "high_similarity_chunks": high_similarity_count,
                "aggregated_score": aggregated_score,
                "plagiarism_flag": plagiarism_flag,
                "document_matches": sorted_matches[:3]
            }
            
        except Exception as e:
            return {"error": f"Similarity check failed: {str(e)}"}

# Initialize processor with lazy loading
@st.cache_resource
def get_processor():
    processor = DocumentProcessor()
    processor.load_index()  # Load index but not model (loaded lazily)
    return processor

def main():
    st.set_page_config(
        page_title="Document Plagiarism Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Automated Document Similarity & Plagiarism Detection")
    st.markdown("**Sprint 2: Enhanced with API Mode Support**")
    st.markdown("---")
    
    # NEW: Mode Detection and Setup
    if BACKEND_URL:
        # API Mode
        api_mode = True
        api_client = APIClient(BACKEND_URL, API_KEY)
    else:
        # Local Mode
        api_mode = False
        processor = get_processor()

    # Enhanced Sidebar with Mode Information
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # NEW: Mode Indicator
        if api_mode:
            st.success(f"🌐 **API Mode**")
            st.info(f"**Backend:** {BACKEND_URL}")
            st.info(f"**API Key:** {API_KEY[:8]}...")
            
            # Backend health check
            if api_client.health_check():
                st.success("✅ Backend Connected")
                
                # Get and display status
                status = api_client.get_status()
                if status:
                    st.metric("📄 Total Documents", status.get("total_docs", 0))
                    st.metric("📝 Total Chunks", status.get("total_chunks", 0))
                    st.info(f"🧠 Backend: {status.get('backend', 'Unknown')}")
                    st.info(f"🤖 Model: {status.get('model', 'Unknown')[:30]}...")
                    total_chunks = status.get("total_chunks", 0)
                    total_docs = status.get("total_docs", 0)
            else:
                st.error("❌ Backend Unavailable")
                st.warning("Check backend is running at the URL above")
                total_chunks = 0
                total_docs = 0
        else:
            st.info("💻 **Local Mode**")
            st.info("Using embedded processor")
            
            # Local mode status
            if processor.index:
                total_chunks = processor.index.ntotal
                total_docs = len(set(
                    chunk.get('doc_id', '') 
                    for chunk in processor.metadata.values() 
                    if isinstance(chunk, dict) and 'doc_id' in chunk
                ))
                st.metric("Documents Indexed", total_docs)
                st.metric("Total Chunks", total_chunks)
                st.metric("Similarity Threshold", f"{SIMILARITY_THRESHOLD:.2f}")
                
                # Show backend type
                backend = "FAISS" if FAISS_AVAILABLE else "Scikit-learn"
                st.info(f"Backend: {backend}")
        
        st.header("System Info")
        st.info("This system detects semantic similarity using embeddings. Results may include false positives/negatives and should not be used as legal proof of plagiarism.")
        
        # Configuration (works in both modes)
        st.header("Configuration")
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.70,
            max_value=0.95,
            value=SIMILARITY_THRESHOLD,
            step=0.05,
            help="Higher = stricter plagiarism detection"
        )
        
        plagiarism_threshold = st.slider(
            "Plagiarism Risk Threshold", 
            min_value=0.1,
            max_value=0.8,
            value=0.3,
            step=0.1,
            help="% of suspicious chunks to flag as HIGH risk"
        )
        
        # Store in session state for use in similarity calculation
        st.session_state['similarity_threshold'] = similarity_threshold
        st.session_state['plagiarism_threshold'] = plagiarism_threshold

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Ingest Documents", "🔍 Check Similarity", "📊 Results"])
    
    with tab1:
        st.header("Document Ingestion")
        st.write("Upload documents to build the similarity index")
        
        uploaded_files = st.file_uploader(
            "Choose files to ingest",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True,
            help=f"Supported: TXT, PDF, DOCX. Max size: {MAX_FILE_SIZE/1024/1024:.1f}MB per file"
        )
        
        if uploaded_files:
            # Quick size check before processing
            oversized_files = []
            for file in uploaded_files:
                try:
                    if len(file.getvalue()) > MAX_FILE_SIZE:
                        oversized_files.append(f"{file.name} ({len(file.getvalue())/1024/1024:.1f}MB)")
                except:
                    pass
            
            if oversized_files:
                st.error(f"Files too large (max {MAX_FILE_SIZE/1024/1024:.1f}MB): {', '.join(oversized_files)}")
            else:
                if st.button("Ingest Documents", type="primary"):
                    # NEW: API vs Local Mode Processing
                    if api_mode:
                        # API Mode
                        with st.spinner("🌐 Processing documents via API..."):
                            result = api_client.ingest_documents(uploaded_files)
                            
                            if result and "results" in result:
                                st.success("✅ Documents ingested successfully!")
                                
                                for res in result["results"]:
                                    if "error" in res:
                                        st.error(f"❌ {res['filename']}: {res['error']}")
                                    else:
                                        st.success(
                                            f"✅ {res['filename']}: "
                                            f"{res['chunks_added']} chunks added "
                                            f"(ID: {res['doc_id'][:12]}...)"
                                        )
                            else:
                                st.error("❌ API ingestion failed")
                    else:
                        # Local Mode (Original Logic)
                        if not processor._model_loaded:
                            with st.spinner("Loading AI model (first time only)..."):
                                processor.load_model()
                        
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            with st.spinner(f"Processing {uploaded_file.name}..."):
                                result = processor.ingest_document(uploaded_file)
                                results.append(result)
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        st.success("Ingestion complete!")
                        for result in results:
                            if result["success"]:
                                st.success(f"✅ {result['filename']}: {result['chunks_added']} chunks added")
                            else:
                                st.error(f"❌ {result.get('filename', 'Unknown')}: {result['error']}")
    
    with tab2:
        st.header("Similarity Check")
        st.write("Upload a document to check for similarity with indexed documents")
        
        # Check if system has any documents
        has_documents = False
        if api_mode:
            # For API mode, we'll check this when button is clicked
            has_documents = True  # Assume true for now, will be checked in API call
        else:
            has_documents = processor.index.ntotal > 0
        
        if not has_documents and not api_mode:
            st.warning("⚠️ No documents in index. Please ingest documents first.")
        else:
            query_file = st.file_uploader(
                "Choose file to check",
                type=['txt', 'pdf', 'docx'],
                help=f"Upload document to check. Max size: {MAX_FILE_SIZE/1024/1024:.1f}MB",
                key="similarity_check"
            )
            
            if query_file:
                # Quick size check
                try:
                    if len(query_file.getvalue()) > MAX_FILE_SIZE:
                        st.error(f"File too large ({len(query_file.getvalue())/1024/1024:.1f}MB). Max size: {MAX_FILE_SIZE/1024/1024:.1f}MB")
                    else:
                        if st.button("Check Similarity", type="primary"):
                            # NEW: API vs Local Mode Processing
                            if api_mode:
                                # API Mode
                                with st.spinner("🌐 Analyzing document via API..."):
                                    results = api_client.check_similarity(query_file)
                                    
                                    if results and "error" not in results:
                                        st.session_state['last_results'] = results
                                        st.success("✅ Analysis complete! Check the Results tab.")
                                    elif results and "error" in results:
                                        st.error(f"❌ {results['error']}")
                                    else:
                                        st.error("❌ API similarity check failed")
                            else:
                                # Local Mode (Original Logic)
                                if not processor._model_loaded:
                                    with st.spinner("Loading AI model..."):
                                        processor.load_model()
                                
                                with st.spinner("Analyzing document..."):
                                    results = processor.check_similarity(query_file)
                                
                                if "error" in results:
                                    st.error(results["error"])
                                else:
                                    st.session_state['last_results'] = results
                                    st.success("✅ Analysis complete! Check the Results tab.")
                except:
                    if st.button("Check Similarity", type="primary"):
                        # Fallback for file access issues
                        if api_mode:
                            with st.spinner("🌐 Analyzing document via API..."):
                                results = api_client.check_similarity(query_file)
                                if results and "error" not in results:
                                    st.session_state['last_results'] = results
                                    st.success("✅ Analysis complete! Check the Results tab.")
                                else:
                                    st.error("❌ Analysis failed")
                        else:
                            if not processor._model_loaded:
                                with st.spinner("Loading AI model..."):
                                    processor.load_model()
                            
                            with st.spinner("Analyzing document..."):
                                results = processor.check_similarity(query_file)
                            
                            if "error" in results:
                                st.error(results["error"])
                            else:
                                st.session_state['last_results'] = results
                                st.success("✅ Analysis complete! Check the Results tab.")

    with tab3:
        st.header("Analysis Results")
        
        if 'last_results' not in st.session_state:
            st.info("No analysis results yet. Run a similarity check first.")
            
            # Show sample documents for testing
            st.subheader("📚 Sample Documents")
            st.markdown("Use these sample documents to test the system:")
            
            sample_docs = [
                ("original.txt", "Baseline document for similarity testing"),
                ("similar.txt", "Paraphrased version - expect 85-95% similarity"), 
                ("different.txt", "Control document - expect <15% similarity")
            ]
            
            for filename, description in sample_docs:
                filepath = f"sample_docs/{filename}"
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            f"📄 {filename}",
                            f.read(),
                            filename,
                            help=description
                        )
        else:
            results = st.session_state['last_results']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Document", results['query_filename'])
            
            with col2:
                score = results.get('aggregated_score', 0)
                st.metric("Similarity Score", f"{score:.2%}")
            
            with col3:
                risk_color = "inverse" if results['plagiarism_flag'] else "normal"
                st.metric(
                    "Plagiarism Risk",
                    "HIGH" if results['plagiarism_flag'] else "LOW",
                    delta_color=risk_color
                )
            
            with col4:
                st.metric(
                    "Suspicious Chunks",
                    f"{results['high_similarity_chunks']}/{results['total_query_chunks']}"
                )
            
            # Enhanced alert messages
            if results['plagiarism_flag']:
                st.error(
                    f"🚨 **HIGH SIMILARITY DETECTED!** "
                    f"This document shows {results['aggregated_score']:.1%} similarity to indexed content. "
                    f"Manual review recommended."
                )
            else:
                st.success(
                    f"✅ **LOW SIMILARITY** "
                    f"This document shows {results['aggregated_score']:.1%} similarity to indexed content. "
                    f"Appears to be original content."
                )
            
            st.markdown("---")
            
            # Detailed matches
            if results['document_matches']:
                st.subheader("Most Similar Documents")
                
                for i, doc_match in enumerate(results['document_matches']):
                    with st.expander(f"📄 {doc_match['filename']} (Max Score: {doc_match['max_score']:.3f})"):
                        st.write(f"**Document ID:** {doc_match['doc_id']}")
                        st.write(f"**High Similarity Matches:** {doc_match['high_similarity_matches']}")
                        
                        if doc_match['matches']:
                            st.write("**Top Matching Chunks:**")
                            for j, match in enumerate(doc_match['matches'][:3]):
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    st.write("**Query Text:**")
                                    st.text_area(
                                        "Query",
                                        match['query_text'],
                                        height=100,
                                        key=f"query_{i}_{j}_{match['chunk_id']}",
                                        label_visibility="collapsed"
                                    )
                                
                                with col_b:
                                    st.write(f"**Matched Text (Score: {match['score']:.3f}):**")
                                    st.text_area(
                                        "Match",
                                        match['matched_text'],
                                        height=100,
                                        key=f"match_{i}_{j}_{match['chunk_id']}",
                                        label_visibility="collapsed"
                                    )
                                
                                # Use dynamic threshold
                                dynamic_threshold = st.session_state.get('similarity_threshold', SIMILARITY_THRESHOLD)
                                if match['score'] >= dynamic_threshold:
                                    st.error("🚨 High similarity detected!")
                                elif match['score'] >= 0.75:
                                    st.warning("⚠️ Moderate similarity")
                                
                                st.markdown("---")
            else:
                st.info("No similar documents found.")
    
    # Enhanced Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔍 AI Plagiarism Detector v2.0**")
    with col2:
        st.markdown("**🤖 Powered by Sentence Transformers**")
    with col3:
        if api_mode:
            st.markdown(f"**🌐 API Mode:** {BACKEND_URL.split('://')[1]}")
        else:
            st.markdown("**💻 Local Mode**")

if __name__ == "__main__":
    main()
