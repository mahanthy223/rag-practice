import numpy as np
import uuid
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.model_name = model_name
        self._load_model()
        
    def _load_model(self):
        print(f"Loading model '{self.model_name}'...")
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            
    def generate_embeddings(self, documents: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model is not loaded. Please check the model name or installation.")
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
embedding_manager = Embedding()
    