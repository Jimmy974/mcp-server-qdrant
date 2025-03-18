"""
Alpine Linux compatible embedding provider.
This module provides an embedding implementation that works well on Alpine Linux
with minimal dependencies and no PyTorch requirement.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

# Configure logging
logger = logging.getLogger(__name__)

class AlpineEmbedProvider(EmbeddingProvider):
    """
    Alpine-optimized embedding provider that works without PyTorch.
    This provider will attempt to use fastembed first, falling back to other
    lightweight options if necessary.
    
    :param model_name: The name of the embedding model to use.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_model = None
        self.vector_size = 384  # Default size for small models
        
        # Try to initialize fastembed
        try:
            from fastembed import TextEmbedding
            logger.info(f"Initializing FastEmbed with model {model_name}")
            self.embedding_model = TextEmbedding(model_name)
            self.embedding_type = "fastembed"
            logger.info("Successfully initialized FastEmbed")
        except Exception as e:
            logger.warning(f"Failed to initialize FastEmbed: {e}")
            logger.warning("Will attempt to use alternative lightweight embedding")
            self._init_fallback_embedding()
    
    def _init_fallback_embedding(self):
        """Initialize a fallback embedding model if fastembed fails."""
        try:
            # Try to import onnxruntime directly
            import onnxruntime as ort
            import numpy as np
            from urllib.request import urlretrieve
            import os
            
            # Use a small, pre-quantized ONNX model
            model_url = "https://huggingface.co/optimum/all-MiniLM-L6-v2-onnx/resolve/main/model_quantized.onnx"
            model_path = "/tmp/model_quantized.onnx"
            
            if not os.path.exists(model_path):
                logger.info(f"Downloading ONNX model to {model_path}")
                urlretrieve(model_url, model_path)
            
            logger.info("Initializing ONNX runtime session")
            self.ort_session = ort.InferenceSession(model_path)
            self.embedding_type = "onnx"
            logger.info("Successfully initialized ONNX fallback model")
            
            # Set more specific configuration
            self.tokenizer = None
            try:
                # Try to load tokenizer if available
                from tokenizers import Tokenizer
                tokenizer_url = "https://huggingface.co/optimum/all-MiniLM-L6-v2-onnx/resolve/main/tokenizer.json"
                tokenizer_path = "/tmp/tokenizer.json"
                
                if not os.path.exists(tokenizer_path):
                    urlretrieve(tokenizer_url, tokenizer_path)
                
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
                logger.info("Successfully loaded tokenizer")
            except Exception as te:
                logger.warning(f"Could not load tokenizer: {te}")
                logger.warning("Will use a basic tokenization method")
        
        except Exception as e:
            logger.error(f"Failed to initialize any embedding model: {e}")
            raise RuntimeError(f"Could not initialize any embedding model: {e}")

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        if not documents:
            return []
            
        if self.embedding_type == "fastembed":
            # Run in a thread pool since FastEmbed is synchronous
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: list(self.embedding_model.embed(documents))
            )
            return [embedding.tolist() for embedding in embeddings]
        
        elif self.embedding_type == "onnx":
            # Use ONNX runtime for inference
            embeddings = []
            for doc in documents:
                embedding = await self._embed_with_onnx(doc)
                embeddings.append(embedding)
            return embeddings
        
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        if not query or not query.strip():
            # Return a zero vector of appropriate dimension to avoid errors
            return [0.0] * self.vector_size
            
        if self.embedding_type == "fastembed":
            # Run in a thread pool since FastEmbed is synchronous
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: list(self.embedding_model.embed([query]))
            )
            return embeddings[0].tolist()
        
        elif self.embedding_type == "onnx":
            # Use ONNX runtime for inference
            return await self._embed_with_onnx(query)
        
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
    
    async def _embed_with_onnx(self, text: str) -> List[float]:
        """Use ONNX runtime to get embeddings."""
        import numpy as np
        
        # Implement basic tokenization if no tokenizer available
        if self.tokenizer:
            # Use proper tokenizer
            encoding = self.tokenizer.encode(text)
            input_ids = encoding.ids
            attention_mask = encoding.attention_mask
            token_type_ids = [0] * len(input_ids)
            
            # Truncate if too long
            max_length = 512
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                token_type_ids = token_type_ids[:max_length]
                
            # Convert to numpy arrays
            input_ids = np.array([input_ids], dtype=np.int64)
            attention_mask = np.array([attention_mask], dtype=np.int64)
            token_type_ids = np.array([token_type_ids], dtype=np.int64)
        else:
            # Very basic tokenization as fallback
            # This is a simplified version and won't work as well as a proper tokenizer
            words = text.lower().split()[:512]  # Simple splitting and truncation
            
            # Create dummy input tensors (this is a simplified approach)
            input_ids = np.array([[i + 1 for i in range(len(words))]], dtype=np.int64)
            attention_mask = np.array([[1] * len(words)], dtype=np.int64)
            token_type_ids = np.array([[0] * len(words)], dtype=np.int64)
        
        # Run inference
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.ort_session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }
            )
        )
        
        # Process the output
        embeddings = outputs[0]
        # Mean pooling
        mask = attention_mask.reshape(-1, attention_mask.shape[-1])
        mask_expanded = np.expand_dims(mask, axis=-1)
        sum_embeddings = np.sum(embeddings * mask_expanded, axis=1)
        sum_mask = np.sum(mask, axis=1, keepdims=True)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        mean_embeddings = sum_embeddings / sum_mask
        
        # Normalize the vector
        norm = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
        normalized_embeddings = mean_embeddings / norm
        
        return normalized_embeddings[0].tolist()

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        """
        model_name = self.model_name.split("/")[-1].lower()
        return f"alpine-{model_name}" 