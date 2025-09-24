"""
FastAPI backend for AI Plagiarism Detection System
Provides REST API endpoints with API key authentication
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import os
import logging
from contextlib import asynccontextmanager

from apps.backend.processor import DocumentProcessor

# Configuration
API_KEY = os.getenv("API_KEY", "demo-secret")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global processor instance
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    global processor
    logger.info("Starting AI Plagiarism Detection Backend...")
    processor = DocumentProcessor()
    logger.info("Backend initialized successfully")
    yield
    logger.info("Shutting down backend...")

app = FastAPI(
    title="AI Plagiarism Detection API",
    description="Production-ready AI system for document similarity detection",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Verify API key from X-API-KEY header or Authorization Bearer token"""
    api_key = None
    
    if x_api_key:
        api_key = x_api_key
    elif authorization and authorization.scheme.lower() == "bearer":
        api_key = authorization.credentials
    
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid API key. Use X-API-KEY header or Authorization: Bearer token."
        )
    return True

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Plagiarism Detection API v2.0.0",
        "status": "operational",
        "endpoints": ["/health", "/status", "/ingest", "/check"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint - no auth required"""
    return {
        "status": "healthy",
        "service": "ai-plagiarism-detector",
        "version": "2.0.0"
    }

@app.get("/status")
async def get_status(auth: bool = Depends(verify_api_key)):
    """Get system status"""
    try:
        status = processor.get_status()
        return status
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.post("/ingest")
async def ingest_documents(
    files: List[UploadFile] = File(...),
    auth: bool = Depends(verify_api_key)
):
    """Ingest multiple documents for similarity indexing"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    
    for file in files:
        try:
            # Read file content
            content = await file.read()
            
            # Process with DocumentProcessor
            result = processor.ingest(file.filename, content)
            
            if result["success"]:
                results.append({
                    "filename": file.filename,
                    "doc_id": result["doc_id"],
                    "chunks_added": result["chunks_added"],
                    "start_index": result["start_index"]
                })
            else:
                results.append({
                    "filename": file.filename,
                    "error": result["error"]
                })
                
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": f"Processing failed: {str(e)}"
            })
    
    return {"results": results}

@app.post("/check")
async def check_similarity(
    file: UploadFile = File(...),
    auth: bool = Depends(verify_api_key)
):
    """Check document similarity against indexed documents"""
    try:
        # Read file content
        content = await file.read()
        
        # Process with DocumentProcessor
        result = processor.check(file.filename, content)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking similarity for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "apps.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
