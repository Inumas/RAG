import os
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

def get_vectorstore():
    """Returns the Chroma vectorstore."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="the_batch_articles"
    )

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
