"""
Refactored DocumentProcessor for FastAPI backend.
Reuses Sprint-1 logic but removes Streamlit UI calls.
"""
import numpy as np
import json
import os
import uuid
import mimetypes
import tempfile
from typing import List, Dict, Any
import re
import logging

# Try FAISS, fallback to scikit-learn if FAISS fails on Windows
FAISS_AVAILABLE = True
try:
    import faiss
except ImportError:
    FAISS_AVAILABLE = False
    from sklearn.neighbors import NearestNeighbors

from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document

# Configuration from Sprint 1 - preserved exactly
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.88"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
OVERLAP = int(os.getenv("OVERLAP", "50"))
DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "5000000"))  # 5MB
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            if len(self._vectors) > 0:
                self.index.fit(self._vectors)
                self._fitted = True

    def search(self, query_vector, k=5):
        if FAISS_AVAILABLE:
            return self.index.search(query_vector, k)
        else:
            if not self._fitted or len(self._vectors) == 0:
                return np.array([[0.0] * k]), np.array([[-1] * k])
            
            k = min(k, len(self._vectors))
            distances, indices = self.index.kneighbors(query_vector, n_neighbors=k)
            scores = 1 - distances
            return scores, indices

def _get_file_ext(filename: str) -> str:
    """Robust file extension detection"""
    name = filename.lower()
    if name.endswith('.pdf'):
        return 'pdf'
    if name.endswith('.docx') or name.endswith('.doc'):
        return 'docx'
    if name.endswith('.txt'):
        return 'txt'
    
    mime, _ = mimetypes.guess_type(name)
    if mime:
        if 'pdf' in mime:
            return 'pdf'
        elif 'word' in mime or 'document' in mime:
            return 'docx'
        elif 'text' in mime:
            return 'txt'
    
    return 'txt'  # Default fallback

class DocumentProcessor:
    """Backend document processor without UI dependencies"""
    
    def __init__(self):
        self.model = None
        self.index = None
        self.metadata = {}
        self._model_loaded = False
        self.load_index()

    def load_model(self):
        """Load the sentence transformer model lazily"""
        if not self._model_loaded:
            try:
                logger.info(f"Loading model: {MODEL_NAME}")
                self.model = SentenceTransformer(MODEL_NAME)
                self._model_loaded = True
                logger.info("Model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load model {MODEL_NAME}: {e}")

    def extract_text(self, filename: str, file_bytes: bytes) -> str:
        """Extract text from file bytes"""
        if len(file_bytes) > MAX_FILE_SIZE:
            raise ValueError(f"File too large ({len(file_bytes)/1024/1024:.1f}MB). Max size: {MAX_FILE_SIZE/1024/1024:.1f}MB")

        file_type = _get_file_ext(filename)
        
        if file_type == 'txt':
            try:
                return file_bytes.decode('utf-8', errors='ignore')
            except Exception as e:
                raise ValueError(f"Error reading text file: {e}")
                
        elif file_type == 'pdf':
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_bytes)
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
                raise ValueError(f"Error reading PDF: {e}")
                
        elif file_type == 'docx':
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_file.flush()
                    
                    doc = Document(tmp_file.name)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    
                    os.unlink(tmp_file.name)
                    return text
            except Exception as e:
                raise ValueError(f"Error reading DOCX: {e}")
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Improved text chunking with proper character position tracking"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        char_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            current_words = len(current_chunk.split())
            sentence_words = len(sentence.split())
            
            if current_words + sentence_words > CHUNK_SIZE and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_pos': char_pos - len(current_chunk),
                    'end_pos': char_pos
                })
                
                words = current_chunk.split()
                overlap_words = words[-OVERLAP:] if len(words) > OVERLAP else words
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence
            
            char_pos += len(sentence) + 1
        
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
                            if len(vectors) > 0:
                                self.index.index.fit(vectors)
                                self.index._fitted = True
                        except Exception as e:
                            logger.warning(f"Could not load sklearn vectors: {e}")
                            self.create_new_index()
                            return
                    else:
                        self.create_new_index()
                        return
                
                with open(METADATA_FILE, 'r') as f:
                    self.metadata = json.load(f)
                
                # Ensure _next_index exists and is consistent
                if "_next_index" not in self.metadata:
                    self._repair_next_index()
                
                # Validate consistency
                numeric_keys = [k for k in self.metadata.keys() if k.isdigit()]
                if self.index.ntotal != len(numeric_keys):
                    logger.warning(f"Index/metadata size mismatch: index={self.index.ntotal}, metadata={len(numeric_keys)}")
                
                total_docs = len(set(chunk.get('doc_id', '') for chunk in self.metadata.values() if isinstance(chunk, dict)))
                logger.info(f"Loaded index: {self.index.ntotal} chunks from {total_docs} documents")
                
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")
                self.create_new_index()
        else:
            self.create_new_index()

    def create_new_index(self):
        """Create new index"""
        dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = IndexWrapper(dimension)
        self.metadata = {"_next_doc_id": 0, "_next_index": 0}
        logger.info("Created new index")

    def _repair_next_index(self):
        """Repair missing _next_index from numeric keys"""
        numeric_keys = [int(k) for k in self.metadata.keys() if k.isdigit()]
        if numeric_keys:
            self.metadata["_next_index"] = max(numeric_keys) + 1
        else:
            self.metadata["_next_index"] = 0
        logger.info(f"Repaired _next_index to {self.metadata['_next_index']}")

    def save_index(self):
        """Save index and metadata atomically"""
        os.makedirs(DATA_DIR, exist_ok=True)
        
        try:
            if FAISS_AVAILABLE and hasattr(self.index, 'index') and hasattr(self.index.index, 'ntotal'):
                faiss.write_index(self.index.index, INDEX_FILE)
            elif not FAISS_AVAILABLE and hasattr(self.index, '_vectors') and len(self.index._vectors) > 0:
                # Save numpy vectors for sklearn fallback
                np.save(INDEX_FILE + ".npy", self.index._vectors)
            
            # Write metadata atomically
            temp_metadata = METADATA_FILE + ".tmp"
            with open(temp_metadata, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            os.replace(temp_metadata, METADATA_FILE)
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def ingest(self, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """Ingest document with robust error handling"""
        try:
            text = self.extract_text(filename, file_bytes)
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

            # Update metadata atomically
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                self.metadata[str(start_idx + i)] = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "filename": filename,
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
                "start_index": start_idx
            }
            
        except Exception as e:
            logger.error(f"Ingestion failed for {filename}: {e}")
            return {"success": False, "error": f"Processing failed: {str(e)}"}

    def check(self, filename: str, file_bytes: bytes, top_k: int = 5) -> Dict[str, Any]:
        """Check similarity with enhanced error handling"""
        try:
            if self.index.ntotal == 0:
                return {"error": "No documents in index. Please ingest documents first."}

            text = self.extract_text(filename, file_bytes)
            if not text:
                return {"error": "Could not extract text from query document"}

            chunks = self.chunk_text(text)
            if not chunks:
                return {"error": "Could not create text chunks from query document"}

            chunk_texts = [chunk['text'] for chunk in chunks]
            query_embeddings = self.create_embeddings(chunk_texts)

            # Search for each query chunk
            all_matches = []
            high_similarity_count = 0

            for i, query_embedding in enumerate(query_embeddings):
                scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and str(idx) in self.metadata:
                        metadata = self.metadata[str(idx)]
                        if isinstance(metadata, dict) and 'doc_id' in metadata:
                            if score >= SIMILARITY_THRESHOLD:
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
                if match["score"] >= SIMILARITY_THRESHOLD:
                    doc_matches[doc_id]["high_similarity_matches"] += 1

            # Sort by similarity
            sorted_matches = sorted(doc_matches.values(), key=lambda x: x["max_score"], reverse=True)

            # Calculate final scores
            total_chunks = len(chunks)
            aggregated_score = high_similarity_count / total_chunks if total_chunks > 0 else 0
            plagiarism_flag = aggregated_score > 0.3

            return {
                "query_filename": filename,
                "total_query_chunks": total_chunks,
                "high_similarity_chunks": high_similarity_count,
                "aggregated_score": aggregated_score,
                "plagiarism_flag": plagiarism_flag,
                "document_matches": sorted_matches[:3]
            }
            
        except Exception as e:
            logger.error(f"Similarity check failed for {filename}: {e}")
            return {"error": f"Similarity check failed: {str(e)}"}

    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        total_docs = len(set(
            chunk.get('doc_id', '') 
            for chunk in self.metadata.values() 
            if isinstance(chunk, dict) and 'doc_id' in chunk
        ))
        
        backend = "FAISS" if FAISS_AVAILABLE else "scikit-learn"
        
        return {
            "total_chunks": self.index.ntotal if self.index else 0,
            "total_docs": total_docs,
            "backend": backend,
            "model": MODEL_NAME,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP
        }
