"""
CLIP Embeddings Module

Provides CLIP-based image and text embeddings using OpenCLIP.
Used for multimodal retrieval (text→image search).

Model: ViT-B/32 (512-dim embeddings)
Device: CPU (no GPU required)
"""

import os
import logging
from typing import Optional, List
import hashlib
import requests
from io import BytesIO

import torch
import open_clip
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"  # Widely available checkpoint
CLIP_EMBEDDING_DIM = 512

# Cache directory for downloaded images
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".clip_cache")


class CLIPEmbedder:
    """
    Singleton class for CLIP embedding computation.
    Lazy-loads the model to save memory if not used.
    """
    
    _instance = None
    _model = None
    _preprocess = None
    _tokenizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _ensure_model_loaded(self):
        """Lazy load the CLIP model."""
        if self._model is None:
            logger.info(f"Loading OpenCLIP model: {CLIP_MODEL_NAME}...")
            
            # Force CPU
            device = "cpu"
            
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME,
                pretrained=CLIP_PRETRAINED,
                device=device
            )
            self._tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
            self._model.eval()
            
            logger.info(f"OpenCLIP model loaded on {device}")
    
    def embed_image(self, image_url: str) -> Optional[List[float]]:
        """
        Compute CLIP embedding for an image from URL.
        
        Args:
            image_url: URL of the image to embed
            
        Returns:
            512-dim embedding vector, or None if failed
        """
        self._ensure_model_loaded()
        
        try:
            # Check cache first
            cache_path = self._get_cache_path(image_url)
            
            if os.path.exists(cache_path):
                # Load from cache
                image = Image.open(cache_path).convert("RGB")
            else:
                # Download image
                response = requests.get(image_url, timeout=10, headers={
                    "User-Agent": "Mozilla/5.0"
                })
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content)).convert("RGB")
                
                # Save to cache
                os.makedirs(CACHE_DIR, exist_ok=True)
                image.save(cache_path, "JPEG")
            
            # Preprocess and embed
            image_tensor = self._preprocess(image).unsqueeze(0)
            
            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.squeeze().tolist()
            
        except Exception as e:
            logger.warning(f"Failed to embed image {image_url}: {e}")
            return None
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Compute CLIP text embedding.
        
        Args:
            text: Query text to embed
            
        Returns:
            512-dim embedding vector, or None if failed
        """
        self._ensure_model_loaded()
        
        try:
            text_tokens = self._tokenizer([text])
            
            with torch.no_grad():
                text_features = self._model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.squeeze().tolist()
            
        except Exception as e:
            logger.warning(f"Failed to embed text: {e}")
            return None
    
    def _get_cache_path(self, url: str) -> str:
        """Generate cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(CACHE_DIR, f"{url_hash}.jpg")


# Convenience functions
def get_clip_embedder() -> CLIPEmbedder:
    """Get the singleton CLIP embedder instance."""
    return CLIPEmbedder()


def embed_image(image_url: str) -> Optional[List[float]]:
    """Embed an image from URL using CLIP."""
    return get_clip_embedder().embed_image(image_url)


def embed_text_clip(text: str) -> Optional[List[float]]:
    """Embed text using CLIP (for image search)."""
    return get_clip_embedder().embed_text(text)


# For testing
if __name__ == "__main__":
    # Test with a sample image
    test_url = "https://charonhub.deeplearning.ai/content/images/2024/04/unnamed---2024-04-17T155856.845-1.png"
    
    embedder = get_clip_embedder()
    
    print("Testing image embedding...")
    img_emb = embedder.embed_image(test_url)
    if img_emb:
        print(f"  Image embedding dim: {len(img_emb)}")
        print(f"  First 5 values: {img_emb[:5]}")
    else:
        print("  Failed!")
    
    print("\nTesting text embedding...")
    text_emb = embedder.embed_text("a diagram of a neural network")
    if text_emb:
        print(f"  Text embedding dim: {len(text_emb)}")
        print(f"  First 5 values: {text_emb[:5]}")
    else:
        print("  Failed!")
