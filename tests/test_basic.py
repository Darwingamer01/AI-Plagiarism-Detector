import pytest
import sys
import os
import tempfile
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_similarity_threshold():
    """Test that similarity threshold is configurable"""
    from apps.frontend.demo_streamlit import SIMILARITY_THRESHOLD
    assert SIMILARITY_THRESHOLD == 0.88

def test_chunk_size():
    """Test chunk size configuration"""
    from apps.frontend.demo_streamlit import CHUNK_SIZE
    assert CHUNK_SIZE == 300

class MockUploadedFile:
    """Mock Streamlit uploaded file for testing"""
    def __init__(self, content: str, name: str):
        self.content = content.encode('utf-8')
        self.name = name
        self.type = "text/plain"
    
    def read(self):
        return self.content
    
    def getvalue(self):
        return self.content

def test_document_processor_initialization():
    """Test that DocumentProcessor initializes correctly"""
    from apps.frontend.demo_streamlit import DocumentProcessor
    
    processor = DocumentProcessor()
    processor.create_new_index()
    
    # Should have empty index and metadata
    assert processor.index.ntotal == 0
    assert "_next_doc_id" in processor.metadata
    assert processor.metadata["_next_doc_id"] == 0

def test_text_chunking():
    """Test that text chunking works correctly"""
    from apps.frontend.demo_streamlit import DocumentProcessor
    
    processor = DocumentProcessor()
    
    # Test with longer text
    long_text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. 
    These systems can automatically improve their performance on a specific task through experience. 
    Deep learning represents a significant advancement in machine learning techniques. 
    Neural networks with multiple layers can learn complex patterns from large datasets. 
    Natural language processing enables computers to understand human language effectively.
    Computer vision allows machines to interpret visual information from digital images.
    """
    
    chunks = processor.chunk_text(long_text.strip())
    
    # Should create at least one chunk
    assert len(chunks) > 0
    
    # Each chunk should have required fields
    for chunk in chunks:
        assert 'text' in chunk
        assert 'start_pos' in chunk
        assert 'end_pos' in chunk
        assert len(chunk['text']) > 10  # Should have meaningful content

def test_end_to_end_similarity_detection():
    """Integration test: ingest document and check similarity with longer, more similar text"""
    from apps.frontend.demo_streamlit import DocumentProcessor
    
    # Create processor
    processor = DocumentProcessor()
    processor.create_new_index()  # Start fresh
    
    # Create longer, more detailed test documents with clear similarities
    original_text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models 
    that enable computer systems to automatically improve their performance on a specific task through experience. 
    Deep learning represents a significant advancement in machine learning techniques by using neural networks 
    with multiple layers to learn complex patterns from large datasets. Natural language processing is another 
    important field that enables computers to understand and generate human language effectively.
    """
    
    # Very similar text with paraphrasing
    similar_text = """
    Machine learning represents a branch of artificial intelligence that concentrates on algorithms and 
    statistical models allowing computer systems to automatically enhance their performance on specific 
    tasks through experience. Deep learning constitutes a major advancement in ML techniques by utilizing 
    neural networks with many layers to learn complex patterns from big datasets. Natural language 
    processing is a crucial area that allows computers to comprehend and produce human language effectively.
    """
    
    original_file = MockUploadedFile(original_text.strip(), "original.txt")
    similar_file = MockUploadedFile(similar_text.strip(), "similar.txt")
    
    # Test ingestion
    result = processor.ingest_document(original_file)
    
    # Debug output
    print(f"Ingestion result: {result}")
    
    assert result["success"] == True
    assert result["chunks_added"] > 0
    assert processor.index.ntotal > 0
    
    # Test similarity check
    similarity_result = processor.check_similarity(similar_file)
    
    # Debug output
    print(f"Similarity result: {similarity_result}")
    
    assert "error" not in similarity_result
    assert "aggregated_score" in similarity_result
    
    # Based on debug results, we expect high similarity (0.79-0.97 range)
    assert similarity_result["aggregated_score"] >= 0.5  # Should find significant similarity
    assert similarity_result["total_query_chunks"] > 0
    assert len(similarity_result["document_matches"]) > 0  # Should find at least one match
    
    # Check that we got good similarity scores
    top_match = similarity_result["document_matches"][0]
    assert "max_score" in top_match
    assert top_match["max_score"] > 0.7  # Should have high similarity (based on debug: 0.79-0.97)
    
    print(f"✅ Integration test passed. Max similarity score: {top_match['max_score']:.3f}")
    print(f"✅ Aggregated score: {similarity_result['aggregated_score']:.3f}")

def test_file_type_detection():
    """Test robust file type detection"""
    from apps.frontend.demo_streamlit import _get_file_ext
    
    # Test various file extensions
    txt_file = MockUploadedFile("test content", "document.txt")
    pdf_file = MockUploadedFile("test content", "document.pdf")
    docx_file = MockUploadedFile("test content", "document.docx")
    
    assert _get_file_ext(txt_file) == 'txt'
    assert _get_file_ext(pdf_file) == 'pdf' 
    assert _get_file_ext(docx_file) == 'docx'

if __name__ == "__main__":
    test_end_to_end_similarity_detection()
