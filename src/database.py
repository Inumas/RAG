import os
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import chromadb

load_dotenv()

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

# Collection names
TEXT_COLLECTION = "the_batch_articles"
IMAGE_COLLECTION = "the_batch_images"

# CLIP embedding dimension
CLIP_EMBEDDING_DIM = 512


def get_vectorstore():
    """Returns the Chroma vectorstore for text embeddings."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=TEXT_COLLECTION
    )


def get_image_vectorstore():
    """
    Returns the Chroma collection for CLIP image embeddings.
    Uses raw ChromaDB client since we're storing pre-computed embeddings.
    """
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # Get or create the image collection
    collection = client.get_or_create_collection(
        name=IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity for CLIP
    )
    
    return collection


def index_image_embeddings(image_data: list):
    """
    Index CLIP image embeddings into the image collection.
    
    Args:
        image_data: List of dicts with keys:
            - id: Document ID (links to text collection)
            - embedding: CLIP image embedding (512-dim)
            - metadata: {source, title, image_url, ...}
    """
    if not image_data:
        return
    
    collection = get_image_vectorstore()
    
    ids = [item["id"] for item in image_data]
    embeddings = [item["embedding"] for item in image_data]
    metadatas = [item.get("metadata", {}) for item in image_data]
    
    # Upsert to handle re-indexing
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"Indexed {len(image_data)} image embeddings")


import hashlib

def generate_id(url, index):
    """Generates a stable ID based on URL and chunk index."""
    return hashlib.md5(f"{url}_{index}".encode()).hexdigest()

def index_documents(documents):
    """Indexes documents into ChromaDB with stable IDs."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    vectorstore = get_vectorstore()
    
    print(f"Processing {len(documents)} documents...")
    
    # Process documents one by one to ensure stable IDs (independent of batch)
    for doc in documents:
        splits = text_splitter.split_documents([doc])
        if not splits:
            continue
            
        ids = []
        source = doc.metadata.get("source", "unknown")
        # Include section header in ID if it exists to prevent collisions for same-url chunks
        section = doc.metadata.get("section_header", "")
        
        for i, split in enumerate(splits):
            # Combined key: URL + Section + Index
            unique_key = f"{source}_{section}_{i}"
            ids.append(hashlib.md5(unique_key.encode()).hexdigest())
            
        # Add to vectorstore (upsert logic handled by Chroma via IDs)
        vectorstore.add_documents(documents=splits, ids=ids)
        
    print("Indexing complete.")

def clear_database():
    """Clears the existing database."""
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print("Database cleared.")


def get_image_collection_stats():
    """Get statistics about the image collection."""
    try:
        collection = get_image_vectorstore()
        count = collection.count()
        return {"count": count, "collection": IMAGE_COLLECTION}
    except Exception as e:
        return {"count": 0, "error": str(e)}

