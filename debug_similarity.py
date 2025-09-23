import sys
sys.path.insert(0, '.')

from apps.frontend.demo_streamlit import DocumentProcessor

class MockUploadedFile:
    def __init__(self, content: str, name: str):
        self.content = content.encode('utf-8')
        self.name = name
        self.type = "text/plain"
    
    def read(self):
        return self.content
    
    def getvalue(self):
        return self.content

def debug_similarity():
    processor = DocumentProcessor()
    processor.create_new_index()
    
    # Test texts
    original_text = """Machine learning is a subset of artificial intelligence. Deep learning uses neural networks with multiple layers. Natural language processing enables computers to understand human language."""
    
    similar_text = """ML is a branch of AI systems. Deep learning utilizes neural networks with many layers. NLP allows computers to understand human language."""
    
    original_file = MockUploadedFile(original_text, "original.txt")
    similar_file = MockUploadedFile(similar_text, "similar.txt")
    
    print("=== DEBUGGING SIMILARITY DETECTION ===")
    
    # Check text extraction
    extracted_original = processor.extract_text(original_file)
    print(f"Original text extracted: '{extracted_original[:100]}...'")
    
    # Check chunking
    chunks = processor.chunk_text(extracted_original)
    print(f"Number of chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: '{chunk['text'][:50]}...'")
    
    # Ingest original
    result = processor.ingest_document(original_file)
    print(f"Ingestion result: {result}")
    print(f"Index total after ingestion: {processor.index.ntotal}")
    
    # Check embedding creation manually
    chunk_texts = [chunk['text'] for chunk in chunks]
    if not processor._model_loaded:
        processor.load_model()
    
    embeddings = processor.create_embeddings(chunk_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample embedding norm: {embeddings[0][:5]} (should be ~1.0 after normalization)")
    
    # Test similarity
    similarity_result = processor.check_similarity(similar_file)
    print(f"Similarity result: {similarity_result}")
    
    if "document_matches" in similarity_result and similarity_result["document_matches"]:
        top_match = similarity_result["document_matches"][0]
        print(f"Top match max score: {top_match.get('max_score', 'N/A')}")
        if "matches" in top_match:
            for match in top_match["matches"][:3]:
                print(f"  Match score: {match.get('score', 'N/A'):.3f}")

if __name__ == "__main__":
    debug_similarity()
