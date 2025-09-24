"""
Backend package for AI Plagiarism Detection System
Contains FastAPI application and DocumentProcessor

Components:
- main.py: FastAPI application with REST endpoints
- processor.py: Core document processing and similarity detection
"""

# Safe imports - only import when actually needed
def get_processor():
    """Lazy import of DocumentProcessor"""
    from .processor import DocumentProcessor
    return DocumentProcessor

def get_app():
    """Lazy import of FastAPI app"""  
    from .main import app
    return app

__version__ = "2.0.0"
