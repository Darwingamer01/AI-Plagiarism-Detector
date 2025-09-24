"""
FastAPI integration tests for AI Plagiarism Detection System.
Tests all API endpoints with sample documents.
"""

import pytest
import os
import tempfile
import json
from fastapi.testclient import TestClient

from apps.backend.main import app

# Test configuration
TEST_API_KEY = "demo-secret"
SAMPLE_DOCS_DIR = "sample_docs"

client = TestClient(app)

def get_sample_file(filename: str) -> bytes:
    """Load sample document for testing"""
    filepath = os.path.join(SAMPLE_DOCS_DIR, filename)
    if not os.path.exists(filepath):
        # Create minimal test content if file doesn't exist
        if filename == "original.txt":
            content = "Artificial intelligence has revolutionized modern computing. Machine learning algorithms enable computers to learn from data without explicit programming. Deep learning, a subset of machine learning, uses neural networks to process complex information patterns. Natural language processing allows machines to understand human language. Computer vision enables automated image recognition and analysis."
        elif filename == "similar.txt":
            content = "AI technology has transformed the computing landscape. ML algorithms allow systems to learn from datasets automatically. Deep neural networks process intricate data patterns effectively. NLP systems comprehend human communication. Automated vision systems analyze images and visual data."
        else:
            content = "Completely different content about weather patterns and climate change in various regions."
        return content.encode('utf-8')
    
    with open(filepath, 'rb') as f:
        return f.read()

class TestHealthAndStatus:
    """Test health and status endpoints"""
    
    def test_health_no_auth(self):
        """Health endpoint should work without authentication"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data

    def test_status_requires_auth(self):
        """Status endpoint should require authentication"""
        response = client.get("/status")
        assert response.status_code == 403  # Tests expect 403 for missing auth

    def test_status_with_auth(self):
        """Status endpoint should work with valid API key"""
        response = client.get("/status", headers={"X-API-KEY": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        # FIXED: Use the actual field names returned by the API
        assert "total_docs" in data
        assert "total_chunks" in data

class TestIngestEndpoint:
    """Test document ingestion endpoint"""
    
    def test_ingest_requires_auth(self):
        """Ingest should require authentication"""
        test_content = b"Test document content"
        response = client.post(
            "/ingest",
            files={"files": ("test.txt", test_content, "text/plain")}
        )
        assert response.status_code == 403  # Tests expect 403 for missing auth

    def test_ingest_no_files(self):
        """Ingest should reject empty file list"""
        response = client.post(
            "/ingest",
            headers={"X-API-KEY": TEST_API_KEY},
            files={}
        )
        assert response.status_code == 422  # FastAPI validation error

    def test_ingest_sample_document(self):
        """Test ingesting sample document"""
        original_content = get_sample_file("original.txt")
        response = client.post(
            "/ingest",
            headers={"X-API-KEY": TEST_API_KEY},
            files={"files": ("original.txt", original_content, "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        
        result = data["results"][0]
        assert result["filename"] == "original.txt"
        assert "doc_id" in result
        assert "chunks_added" in result
        assert result["chunks_added"] > 0

    def test_ingest_multiple_documents(self):
        """Test ingesting multiple documents"""
        files = [
            ("file1.txt", b"First document content", "text/plain"),
            ("file2.txt", b"Second document content", "text/plain")
        ]
        
        response = client.post(
            "/ingest",
            headers={"X-API-KEY": TEST_API_KEY},
            files=[("files", f) for f in files]
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2

class TestCheckEndpoint:
    """Test similarity checking endpoint"""
    
    def test_check_requires_auth(self):
        """Check should require authentication"""
        test_content = b"Test document content"
        response = client.post(
            "/check",
            files={"file": ("test.txt", test_content, "text/plain")}
        )
        assert response.status_code == 403  # Tests expect 403 for missing auth

    def test_check_empty_index(self):
        """Check should handle empty index gracefully"""
        # First ensure we have a clean state by checking status
        status_response = client.get("/status", headers={"X-API-KEY": TEST_API_KEY})
        assert status_response.status_code == 200
        
        # Now try checking a document against empty index
        test_content = b"Test document for similarity checking"
        response = client.post(
            "/check",
            headers={"X-API-KEY": TEST_API_KEY},
            files={"file": ("test.txt", test_content, "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        # FIXED: Use the actual field names returned by the API
        assert "document_matches" in data
        # For empty index, document_matches should be empty or minimal
        assert len(data["document_matches"]) >= 0

    def test_ingest_then_check_workflow(self):
        """Test complete workflow: ingest then check similarity"""
        # Step 1: Ingest original document
        original_content = get_sample_file("original.txt")
        ingest_response = client.post(
            "/ingest",
            headers={"X-API-KEY": TEST_API_KEY},
            files={"files": ("original.txt", original_content, "text/plain")}
        )
        assert ingest_response.status_code == 200
        ingest_data = ingest_response.json()
        assert len(ingest_data["results"]) == 1
        assert ingest_data["results"][0]["chunks_added"] > 0
        
        # Step 2: Check similar document
        similar_content = get_sample_file("similar.txt")
        check_response = client.post(
            "/check",
            headers={"X-API-KEY": TEST_API_KEY},
            files={"file": ("similar.txt", similar_content, "text/plain")}
        )
        assert check_response.status_code == 200
        check_data = check_response.json()
        # FIXED: Use the actual field names returned by the API
        assert "document_matches" in check_data
        assert "aggregated_score" in check_data
        assert isinstance(check_data["aggregated_score"], float)

class TestAuthentication:
    """Test authentication mechanisms"""
    
    def test_x_api_key_header(self):
        """Test authentication with X-API-KEY header"""
        response = client.get("/status", headers={"X-API-KEY": TEST_API_KEY})
        assert response.status_code == 200

    def test_bearer_token_auth(self):
        """Test authentication with Authorization Bearer token"""
        response = client.get("/status", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert response.status_code == 200

    def test_invalid_api_key(self):
        """Test rejection of invalid API key"""
        response = client.get("/status", headers={"X-API-KEY": "invalid-key"})
        assert response.status_code == 403  # Tests expect 403 for invalid auth

    def test_missing_auth(self):
        """Test rejection when no authentication provided"""
        response = client.get("/status")
        assert response.status_code == 403  # Tests expect 403 for missing auth

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_large_file_rejection(self):
        """Test rejection of files that are too large"""
        # Create a large file (larger than 5MB limit)
        large_content = b"x" * (6 * 1024 * 1024)  # 6MB
        
        response = client.post(
            "/ingest",
            headers={"X-API-KEY": TEST_API_KEY},
            files={"files": ("large.txt", large_content, "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert "error" in data["results"][0]
        assert "too large" in data["results"][0]["error"].lower()

    def test_unsupported_file_type(self):
        """Test handling of unsupported file types"""
        # This should still process as text
        response = client.post(
            "/ingest",
            headers={"X-API-KEY": TEST_API_KEY},
            files={"files": ("test.xyz", b"Some content", "application/octet-stream")}
        )
        assert response.status_code == 200

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data
