"""
Index CLIP Image Embeddings

Scans existing documents in the text collection, extracts image URLs,
computes CLIP embeddings, and stores them in the image collection.

Usage:
    python scripts/index_images.py [--max N]
"""

import sys
import os
import argparse
import hashlib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database import get_vectorstore, index_image_embeddings, get_image_collection_stats
from clip_embeddings import get_clip_embedder


def get_unique_image_urls() -> dict:
    """
    Get all unique image URLs from the text collection.
    Returns dict mapping image_url -> {doc_id, source, title}
    """
    vs = get_vectorstore()
    results = vs.get()
    
    image_map = {}
    
    for i, metadata in enumerate(results.get('metadatas', [])):
        if not metadata:
            continue
            
        image_url = metadata.get('image_url')
        if not image_url:
            continue
        
        # Use the document ID if available, otherwise generate one
        doc_id = results['ids'][i] if 'ids' in results else hashlib.md5(image_url.encode()).hexdigest()
        
        # Store unique image URLs
        if image_url not in image_map:
            image_map[image_url] = {
                'doc_id': doc_id,
                'source': metadata.get('source', ''),
                'title': metadata.get('title', ''),
                'topic': metadata.get('topic', ''),
                'issue_number': metadata.get('issue_number'),
            }
    
    return image_map


def index_all_images(max_images: int = None, batch_size: int = 10):
    """
    Index all images from the text collection into the image collection.
    
    Args:
        max_images: Maximum number of images to index (None = all)
        batch_size: Number of images to process before saving
    """
    print("Fetching image URLs from text collection...")
    image_map = get_unique_image_urls()
    print(f"Found {len(image_map)} unique images")
    
    if max_images:
        image_map = dict(list(image_map.items())[:max_images])
        print(f"Limiting to {len(image_map)} images")
    
    # Initialize CLIP embedder
    embedder = get_clip_embedder()
    
    # Process in batches
    image_urls = list(image_map.keys())
    total = len(image_urls)
    success = 0
    failed = 0
    
    batch_data = []
    
    for i, url in enumerate(image_urls):
        print(f"[{i+1}/{total}] Processing: {url[:60]}...")
        
        # Compute embedding
        embedding = embedder.embed_image(url)
        
        if embedding:
            info = image_map[url]
            batch_data.append({
                'id': hashlib.md5(url.encode()).hexdigest(),
                'embedding': embedding,
                'metadata': {
                    'image_url': url,
                    'source': info['source'],
                    'title': info['title'],
                    'topic': info['topic'],
                    'issue_number': info['issue_number'],
                    'doc_id': info['doc_id'],  # Link to text collection
                }
            })
            success += 1
        else:
            failed += 1
            print(f"  FAILED: Could not embed image")
        
        # Save batch
        if len(batch_data) >= batch_size:
            index_image_embeddings(batch_data)
            batch_data = []
    
    # Save remaining
    if batch_data:
        index_image_embeddings(batch_data)
    
    print("\n" + "=" * 50)
    print(f"Indexing complete!")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    
    # Show stats
    stats = get_image_collection_stats()
    print(f"  Total images in collection: {stats.get('count', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Index CLIP image embeddings")
    parser.add_argument("--max", type=int, default=None, help="Maximum images to index")
    parser.add_argument("--batch", type=int, default=10, help="Batch size for saving")
    args = parser.parse_args()
    
    index_all_images(max_images=args.max, batch_size=args.batch)


if __name__ == "__main__":
    main()
