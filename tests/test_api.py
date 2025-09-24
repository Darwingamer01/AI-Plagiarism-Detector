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
            content = "This is a test document about artificial intelligence and machine learning."
        elif filename == "similar.txt":
            content = "This document discusses AI and ML technologies in detail."
        else:
            content = "Completely different content about weather patterns."
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
        assert response.status_code == 401

    def test_status_with_auth(self):
        """Status endpoint should work with valid API key"""
        response = client.get("/status", headers={"X-API-KEY": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "total_chunks" in data
        assert "total_docs" in data
        assert "backend" in data
        assert "model" in data

class TestIngestEndpoint:
    """Test document ingestion endpoint"""
    
    def test_ingest_requires_auth(self):
        """Ingest should require authentication"""
        test_content = b"Test document content"
        response = client.post(
            "/ingest",
            files={"files": ("test.txt", test_content, "text/plain")}
        )
        assert response.status_code == 401

    def test_ingest_no_files(self):
        """Ingest should reject empty file list"""
        response = client.post(
            "/ingest",
            headers={"X-API-KEY": TEST_API_KEY}
        )
        assert response.status_code == 422  # Validation error

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
        assert "start_index" in result
        assert result["chunks_added"] > 0

    def test_ingest_multiple_documents(self):
        """Test ingesting multiple documents"""
        original_content = get_sample_file("original.txt")
        different_content = get_sample_file("different.txt")
        
        response = client.post(
            "/ingest",
            headers={"X-API-KEY": TEST_API_KEY},
            files=[
                ("files", ("original.txt", original_content, "text/plain")),
                ("files", ("different.txt", different_content, "text/plain"))
            ]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        
        # Check both files were processed
        filenames = [result["filename"] for result in data["results"]]
        assert "original.txt" in filenames
        assert "different.txt" in filenames

class TestCheckEndpoint:
    """Test similarity check endpoint"""
    
    def test_check_requires_auth(self):
        """Check should require authentication"""
        test_content = b"Test document content"
        response = client.post(
            "/check",
            files={"file": ("test.txt", test_content, "text/plain")}
        )
        assert response.status_code == 401

    def test_check_empty_index(self):
        """Check should handle empty index gracefully"""
        # First ensure we have a clean state by checking status
        status_response = client.get("/status", headers={"X-API-KEY": TEST_API_KEY})
        assert status_response.status_code == 200
        
        similar_content = get_sample_file("similar.txt")
        
        response = client.post(
            "/check",
            headers={"X-API-KEY": TEST_API_KEY},
            files={"file": ("similar.txt", similar_content, "text/plain")}
        )
        
        # Should either work with 0 results or return an error about empty index
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            data = response.json()
            assert "aggregated_score" in data
            assert data["aggregated_score"] == 0

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
        
        # Step 2: Check similarity with similar document
        similar_content = get_sample_file("similar.txt")
        
        check_response = client.post(
            "/check",
            headers={"X-API-KEY": TEST_API_KEY},
            files={"file": ("similar.txt", similar_content, "text/plain")}
        )
        
        assert check_response.status_code == 200
        check_data = check_response.json()
        
        # Verify response structure
        assert "aggregated_score" in check_data
        assert "plagiarism_flag" in check_data
        assert "document_matches" in check_data
        assert "query_filename" in check_data
        assert check_data["query_filename"] == "similar.txt"
        
        # Should find some similarity (depends on sample content)
        assert check_data["aggregated_score"] >= 0
        
        # Check if original.txt appears in matches (if similarity > 0)
        if check_data["aggregated_score"] > 0:
            assert len(check_data["document_matches"]) > 0
            # Check if original.txt is in the matches
            matched_files = [match["filename"] for match in check_data["document_matches"]]
            assert "original.txt" in matched_files

class TestAuthentication:
    """Test API authentication mechanisms"""
    
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
        assert response.status_code == 401

    def test_missing_auth(self):
        """Test rejection when no authentication provided"""
        response = client.get("/status")
        assert response.status_code == 401

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
        # Note: The backend now defaults to 'txt' for unknown types
        unknown_content = b"Binary content that might not be text"
        
        response = client.post(
            "/ingest",
            headers={"X-API-KEY": TEST_API_KEY},
            files={"files": ("unknown.bin", unknown_content, "application/octet-stream")}
        )
        
        assert response.status_code == 200
        # Should either succeed (treated as text) or fail gracefully
        data = response.json()
        assert len(data["results"]) == 1
        # Result should have either success or error, not crash

def test_root_endpoint():
    """Test root endpoint provides API information"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data
    assert isinstance(data["endpoints"], list)
