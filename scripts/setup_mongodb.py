#!/usr/bin/env python3
"""
MongoDB Atlas Setup Script

This script helps you set up MongoDB Atlas Vector Search for the RAG chatbot.
It will:
1. Test your MongoDB connection
2. Create the required collection
3. Guide you through creating the vector search index
4. Optionally migrate data from FAISS

Usage:
    python scripts/setup_mongodb.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print setup banner."""
    print("\n" + "=" * 60)
    print("  🍃 MongoDB Atlas Vector Search Setup")
    print("  Discord RAG FAQ Chatbot")
    print("=" * 60 + "\n")


def check_pymongo():
    """Check if pymongo is installed."""
    try:
        import pymongo
        print(f"✅ pymongo version: {pymongo.version}")
        return True
    except ImportError:
        print("❌ pymongo is not installed")
        print("   Run: pip install 'pymongo[srv]'")
        return False


def get_mongodb_uri():
    """Get MongoDB URI from environment or user input."""
    uri = os.getenv("MONGODB_URI", "")
    
    if uri and "username:password" not in uri:
        return uri
    
    print("\n📝 MongoDB Atlas Connection URI")
    print("-" * 40)
    print("You need a MongoDB Atlas connection string.")
    print("Format: mongodb+srv://<username>:<password>@<cluster>.mongodb.net/")
    print("\nTo get your connection string:")
    print("1. Go to https://cloud.mongodb.com")
    print("2. Select your cluster")
    print("3. Click 'Connect' → 'Drivers'")
    print("4. Copy the connection string")
    print("5. Replace <password> with your actual password\n")
    
    uri = input("Enter your MongoDB URI: ").strip()
    return uri


def test_connection(uri: str) -> bool:
    """Test MongoDB connection."""
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError
    
    print("\n🔌 Testing MongoDB Connection...")
    
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=10000)
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB Atlas!")
        
        # Get cluster info
        server_info = client.server_info()
        print(f"   Server version: {server_info.get('version', 'unknown')}")
        
        return True
        
    except ServerSelectionTimeoutError:
        print("❌ Connection timeout. Check your URI and network.")
        return False
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def setup_collection(uri: str, db_name: str, collection_name: str):
    """Set up the collection."""
    from pymongo import MongoClient
    
    print(f"\n📦 Setting up collection: {db_name}.{collection_name}")
    
    client = MongoClient(uri)
    db = client[db_name]
    
    # Check if collection exists
    if collection_name in db.list_collection_names():
        count = db[collection_name].count_documents({})
        print(f"✅ Collection exists with {count} documents")
    else:
        # Create collection
        db.create_collection(collection_name)
        print(f"✅ Created collection: {collection_name}")
    
    return db[collection_name]


def get_vector_index_definition(dimension: int = 384):
    """Get the vector search index definition."""
    return {
        "name": "vector_index",
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dimension,
                    "similarity": "cosine"
                },
                {
                    "type": "filter",
                    "path": "source"
                }
            ]
        }
    }


def print_index_instructions(index_name: str, dimension: int):
    """Print instructions for creating the vector search index."""
    index_def = get_vector_index_definition(dimension)
    
    print("\n" + "=" * 60)
    print("  📋 Create Vector Search Index")
    print("=" * 60)
    print("""
The vector search index must be created in MongoDB Atlas UI:

1. Go to MongoDB Atlas → Your Cluster → Atlas Search

2. Click "Create Search Index"

3. Select "JSON Editor" 

4. Choose your database and collection:
   - Database: rag_chatbot
   - Collection: knowledge_base

5. Copy and paste this index definition:
""")
    
    # Simpler format for Atlas UI
    atlas_definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dimension,
                "similarity": "cosine"
            },
            {
                "type": "filter",
                "path": "source"
            }
        ]
    }
    
    print(json.dumps(atlas_definition, indent=2))
    
    print(f"""
6. Set index name to: {index_name}

7. Click "Create Search Index"

8. Wait for the index to be "Active" (may take a few minutes)

NOTE: Vector Search requires M10+ cluster tier (not M0 free tier)
""")


def check_vector_index(collection, index_name: str) -> bool:
    """Check if vector index exists."""
    print(f"\n🔍 Checking for vector index: {index_name}")
    
    try:
        # List search indexes
        indexes = list(collection.list_search_indexes())
        
        for idx in indexes:
            if idx.get("name") == index_name:
                status = idx.get("status", "unknown")
                print(f"✅ Vector index found! Status: {status}")
                return status == "READY"
        
        print("⚠️  Vector index not found. Please create it manually.")
        return False
        
    except Exception as e:
        logger.debug(f"Could not list search indexes: {e}")
        print("⚠️  Could not check index status (may require higher privileges)")
        return False


def migrate_from_faiss(uri: str, db_name: str, collection_name: str):
    """Migrate data from FAISS to MongoDB."""
    print("\n📤 Migration from FAISS to MongoDB")
    print("-" * 40)
    
    faiss_path = Path(os.getenv("FAISS_INDEX_PATH", "./data/faiss_index"))
    metadata_path = faiss_path.with_suffix(".json")
    
    if not metadata_path.exists():
        print("ℹ️  No FAISS data found to migrate")
        return 0
    
    # Load FAISS metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    chunks_data = metadata.get("chunks", {})
    if not chunks_data:
        print("ℹ️  FAISS index is empty")
        return 0
    
    print(f"Found {len(chunks_data)} chunks in FAISS")
    
    response = input("Migrate to MongoDB? [y/N]: ").strip().lower()
    if response != 'y':
        print("Migration skipped")
        return 0
    
    # Connect to MongoDB
    from pymongo import MongoClient, UpdateOne
    
    client = MongoClient(uri)
    collection = client[db_name][collection_name]
    
    # Prepare documents
    operations = []
    for idx, chunk_dict in chunks_data.items():
        doc = {
            "_id": chunk_dict["chunk_id"],
            "text": chunk_dict["text"],
            "embedding": chunk_dict.get("embedding"),
            "source": chunk_dict["source"],
            "chunk_index": chunk_dict.get("chunk_index", 0),
            "total_chunks": chunk_dict.get("total_chunks", 0),
            "metadata": chunk_dict.get("metadata", {}),
        }
        operations.append(
            UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
        )
    
    # Execute bulk write
    if operations:
        result = collection.bulk_write(operations)
        migrated = result.upserted_count + result.modified_count
        print(f"✅ Migrated {migrated} chunks to MongoDB")
        return migrated
    
    return 0


def update_env_file():
    """Update .env file to use MongoDB."""
    env_path = project_root / ".env"
    
    print("\n📝 Update Configuration")
    print("-" * 40)
    
    response = input("Switch to MongoDB as default vector store? [y/N]: ").strip().lower()
    if response != 'y':
        print("Configuration unchanged")
        return
    
    # Read current .env
    with open(env_path, "r") as f:
        content = f.read()
    
    # Update provider
    content = content.replace(
        "VECTOR_STORE_PROVIDER=faiss",
        "VECTOR_STORE_PROVIDER=mongodb"
    )
    
    # Write back
    with open(env_path, "w") as f:
        f.write(content)
    
    print("✅ Updated .env to use MongoDB")


def main():
    """Main setup flow."""
    print_banner()
    
    # Step 1: Check pymongo
    if not check_pymongo():
        print("\nPlease install pymongo first:")
        print("  pip install 'pymongo[srv]'")
        return
    
    # Step 2: Get MongoDB URI
    uri = get_mongodb_uri()
    if not uri:
        print("❌ MongoDB URI is required")
        return
    
    # Step 3: Test connection
    if not test_connection(uri):
        print("\n❌ Could not connect to MongoDB. Please check your URI.")
        return
    
    # Step 4: Get configuration
    db_name = os.getenv("MONGODB_DATABASE", "rag_chatbot")
    collection_name = os.getenv("MONGODB_COLLECTION", "knowledge_base")
    index_name = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")
    
    # Step 5: Setup collection
    collection = setup_collection(uri, db_name, collection_name)
    
    # Step 6: Check/create vector index
    index_exists = check_vector_index(collection, index_name)
    
    if not index_exists:
        # Determine dimension
        embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        dimension = 384 if "MiniLM" in embedding_model else 768
        
        print_index_instructions(index_name, dimension)
        
        input("\nPress Enter after creating the index...")
        
        # Check again
        check_vector_index(collection, index_name)
    
    # Step 7: Offer migration
    migrate_from_faiss(uri, db_name, collection_name)
    
    # Step 8: Update configuration
    update_env_file()
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✅ MongoDB Atlas Setup Complete!")
    print("=" * 60)
    print(f"""
Configuration:
  Database:   {db_name}
  Collection: {collection_name}
  Index:      {index_name}

Next Steps:
  1. Ensure your .env has the correct MONGODB_URI
  2. Set VECTOR_STORE_PROVIDER=mongodb in .env
  3. Run the bot: python run_bot.py
  
The bot will now use MongoDB Atlas for vector storage!
""")


if __name__ == "__main__":
    main()
