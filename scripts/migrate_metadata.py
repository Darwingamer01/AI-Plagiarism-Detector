#!/usr/bin/env python3
"""
Metadata migration and repair tool for AI Plagiarism Detection System.
Ensures _next_index consistency and optionally migrates to MongoDB.
"""
import json
import os
import argparse
import logging
from typing import Dict, Any
import numpy as np

# Try to import MongoDB client
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_metadata(metadata_file: str) -> Dict[str, Any]:
    """Load metadata from JSON file"""
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return {}
    
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return {}

def save_metadata(metadata: Dict[str, Any], metadata_file: str):
    """Save metadata to JSON file atomically"""
    try:
        temp_file = metadata_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        os.replace(temp_file, metadata_file)
        logger.info(f"Metadata saved to {metadata_file}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        raise

def repair_next_index(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Repair missing or incorrect _next_index"""
    numeric_keys = [int(k) for k in metadata.keys() if k.isdigit()]
    
    if "_next_index" not in metadata:
        if numeric_keys:
            metadata["_next_index"] = max(numeric_keys) + 1
        else:
            metadata["_next_index"] = 0
        logger.info(f"Added missing _next_index: {metadata['_next_index']}")
    else:
        # Validate existing _next_index
        expected_next_index = max(numeric_keys) + 1 if numeric_keys else 0
        if metadata["_next_index"] != expected_next_index:
            logger.warning(f"Correcting _next_index from {metadata['_next_index']} to {expected_next_index}")
            metadata["_next_index"] = expected_next_index
        else:
            logger.info(f"_next_index is correct: {metadata['_next_index']}")
    
    return metadata

def validate_index_consistency(metadata: Dict[str, Any], index_file: str):
    """Validate metadata and index file consistency"""
    numeric_keys = [k for k in metadata.keys() if k.isdigit()]
    metadata_count = len(numeric_keys)
    
    # Check FAISS index
    if os.path.exists(index_file):
        try:
            import faiss
            index = faiss.read_index(index_file)
            index_count = index.ntotal
            logger.info(f"FAISS index contains {index_count} vectors")
            
            if metadata_count != index_count:
                logger.warning(f"Mismatch: metadata has {metadata_count} entries, FAISS index has {index_count}")
            else:
                logger.info("✅ Metadata and FAISS index are consistent")
                
        except ImportError:
            logger.info("FAISS not available, checking numpy fallback")
        except Exception as e:
            logger.error(f"Failed to read FAISS index: {e}")
    
    # Check numpy fallback
    numpy_file = index_file + ".npy"
    if os.path.exists(numpy_file):
        try:
            vectors = np.load(numpy_file)
            vector_count = vectors.shape[0]
            logger.info(f"Numpy index contains {vector_count} vectors")
            
            if metadata_count != vector_count:
                logger.warning(f"Mismatch: metadata has {metadata_count} entries, numpy index has {vector_count}")
            else:
                logger.info("✅ Metadata and numpy index are consistent")
                
        except Exception as e:
            logger.error(f"Failed to read numpy index: {e}")

def migrate_to_mongodb(metadata: Dict[str, Any], mongo_uri: str, db_name: str = "plagiarism_detector"):
    """Migrate metadata to MongoDB"""
    if not MONGO_AVAILABLE:
        logger.error("pymongo not available. Install with: pip install pymongo")
        return False
    
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        
        # Test connection
        client.admin.command('ping')
        logger.info(f"Connected to MongoDB: {mongo_uri}")
        
        # Create collections
        chunks_collection = db.chunks
        system_collection = db.system
        
        # Migrate chunk data
        chunks_to_insert = []
        for key, value in metadata.items():
            if key.isdigit() and isinstance(value, dict):
                chunk_doc = {
                    "_id": int(key),
                    **value
                }
                chunks_to_insert.append(chunk_doc)
        
        if chunks_to_insert:
            # Use upsert to handle existing data
            for chunk in chunks_to_insert:
                chunks_collection.replace_one(
                    {"_id": chunk["_id"]},
                    chunk,
                    upsert=True
                )
            logger.info(f"Migrated {len(chunks_to_insert)} chunks to MongoDB")
        
        # Migrate system data
        system_data = {k: v for k, v in metadata.items() if not k.isdigit()}
        if system_data:
            system_collection.replace_one(
                {"_id": "config"},
                {"_id": "config", **system_data},
                upsert=True
            )
            logger.info("Migrated system configuration to MongoDB")
        
        # Create indices for better performance
        chunks_collection.create_index("doc_id")
        chunks_collection.create_index("chunk_id")
        
        logger.info("✅ MongoDB migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"MongoDB migration failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Migrate and repair plagiarism detector metadata")
    parser.add_argument("--metadata-file", default="./data/metadata.json", help="Path to metadata.json")
    parser.add_argument("--index-file", default="./data/faiss.index", help="Path to index file")
    parser.add_argument("--mongo-uri", help="MongoDB connection URI for migration")
    parser.add_argument("--repair-only", action="store_true", help="Only repair _next_index, skip MongoDB migration")
    parser.add_argument("--validate-only", action="store_true", help="Only validate consistency, no changes")
    
    args = parser.parse_args()
    
    logger.info("Starting metadata migration/repair...")
    
    # Load metadata
    metadata = load_metadata(args.metadata_file)
    if not metadata:
        logger.error("No metadata to process")
        return 1
    
    # Validate consistency
    validate_index_consistency(metadata, args.index_file)
    
    if args.validate_only:
        logger.info("Validation complete")
        return 0
    
    # Repair _next_index
    original_metadata = metadata.copy()
    metadata = repair_next_index(metadata)
    
    # Save if changed
    if metadata != original_metadata:
        save_metadata(metadata, args.metadata_file)
        logger.info("✅ Metadata repaired and saved")
    else:
        logger.info("✅ No repairs needed")
    
    # MongoDB migration
    if args.mongo_uri and not args.repair_only:
        success = migrate_to_mongodb(metadata, args.mongo_uri)
        if success:
            logger.info("✅ All operations completed successfully")
        else:
            logger.error("❌ MongoDB migration failed")
            return 1
    
    logger.info("✅ Migration/repair completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())
