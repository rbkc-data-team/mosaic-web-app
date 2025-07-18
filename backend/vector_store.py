from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import time
import threading
from datetime import datetime, timedelta

# Global client to ensure in-memory data persistence across function calls
_QDRANT_CLIENT = None
_STORE_CREATION_TIME = None
_CLEANUP_THREAD = None
_CLEANUP_INTERVAL = 1200    #3600  # 1 hour in seconds

def get_qdrant_client():
    """Get or create a singleton QdrantClient instance"""
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        _QDRANT_CLIENT = QdrantClient(":memory:")
        print("Created new QdrantClient singleton instance")
    return _QDRANT_CLIENT

def start_cleanup_timer():
     """Start a timer to clean up the vector store after 20 minutes of inactivity"""
     global _CLEANUP_THREAD, _STORE_CREATION_TIME
     
     def cleanup_after_timeout():
         global _STORE_CREATION_TIME
         while True:
             if _STORE_CREATION_TIME and (datetime.now() - _STORE_CREATION_TIME).total_seconds() >= _CLEANUP_INTERVAL:
                 print("Automatic cleanup triggered after 20 minutes")
                 cleanup_vector_store()
                 _STORE_CREATION_TIME = None
                 break
             time.sleep(60)  # Check every minute
     
     # Stop any existing cleanup thread
     if _CLEANUP_THREAD and _CLEANUP_THREAD.is_alive():
         _CLEANUP_THREAD.join(timeout=1)
     
     # Start new cleanup thread
     _CLEANUP_THREAD = threading.Thread(target=cleanup_after_timeout)
     _CLEANUP_THREAD.daemon = True
     _CLEANUP_THREAD.start()

def create_qdrant_store(collection_name="vector_store"):
    """Create an in-memory Qdrant vector store"""
    global _STORE_CREATION_TIME
    # Use the global client
    client = get_qdrant_client()
    
    # Create a new collection if it doesn't exist
    try:
        client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
    except Exception:
        print(f"Creating new collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    
    # Create store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name
    )
    # Update creation time and start cleanup timer
    _STORE_CREATION_TIME = datetime.now()
    start_cleanup_timer()

    return vector_store

def create_vector_index(documents, vector_store, embed_model):
    """Create vector index from documents"""
    print(f"Creating vector index with {len(documents)} documents")
        
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create the index with explicit embedding
    try:
        # First ensure documents are properly formatted
        formatted_documents = []
        for i, doc in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}")
            if not hasattr(doc, 'get_content'):
                # If it's not already a Document object, create one
                formatted_documents.append(Document(text=str(doc)))
            else:
                formatted_documents.append(doc)
        
        print(f"Creating index with {len(formatted_documents)} formatted documents")
        index = VectorStoreIndex.from_documents(
            formatted_documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        # Verify points were added to collection - use global client directly
        client = get_qdrant_client()
        collection_name = vector_store.collection_name
        count = client.count(collection_name).count
        print(f"Collection {collection_name} now has {count} vectors")
        
        print(f"Vector index created successfully")
        return index
    except Exception as e:
        import traceback
        print(f"Error creating vector index: {e}")
        print(traceback.format_exc())
        raise

def create_chat_engine(llm, collection_name="vector_store"):
    """Create chat engine for querying the vector store directly"""
    # Create the vector store
    vector_store = create_qdrant_store(collection_name)
    
    # Create storage context and index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    
    # Create memory and chat engine
    memory = ChatMemoryBuffer.from_defaults()
    
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        verbose=True,
        similarity_top_k=15,
        vector_store_query_mode="hybrid"
    )
    return chat_engine

def check_collection_status(collection_name="vector_store"):
    """Check if a collection exists and contains vectors"""
    try:
        # Use the global client
        client = get_qdrant_client()
        try:
            collection_info = client.get_collection(collection_name)
            count = client.count(collection_name).count
            print(f"Collection {collection_name} exists with {count} points")
            return count > 0
        except Exception as e:
            print(f"Error checking collection {collection_name}: {e}")
            return False
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return False

def cleanup_vector_store():
    """Clean up the in-memory vector store"""
    global _STORE_CREATION_TIME
    try:
        # Use the global client
        client = get_qdrant_client()
        collections = ["person_store", "group_store", "vector_store", "mosaic_data"]
        
        for collection in collections:
            try:
                # First check if collection exists and has points
                try:
                    count = client.count(collection).count
                    print(f"Collection {collection} has {count} points before deletion")
                except:
                    print(f"Collection {collection} doesn't exist or can't be counted")
                
                # Attempt to delete
                print(f"Attempting to delete collection: {collection}")
                client.delete_collection(collection)
                print(f"Successfully deleted collection: {collection}")
            except Exception as e:
                print(f"Error deleting collection {collection}: {e}")
                
        # Reset creation time
        _STORE_CREATION_TIME = None        
    except Exception as e:
        print(f"Error initializing QdrantClient: {e}") 