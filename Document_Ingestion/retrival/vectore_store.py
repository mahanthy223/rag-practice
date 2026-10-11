import os
import uuid
from pathlib import Path
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import numpy as np

class VectorStore:
    def __init__(
        self,
        collection_name: str = "pdf_collection",
        persist_directory: str = str(
            Path(__file__).resolve().parents[1] / "Documents" / "vectore_store"
        ),
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_chromadb()
        
    def _initialize_chromadb(self):
        print("Initializing ChromaDB client...")
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"description": "Collection of PDF document embeddings"})
            print(f"ChromaDB client initialized. Collection '{self.collection_name}' is ready.")
        except Exception as e:
            print(f"Error initializing ChromaDB client: {e}")
            
    def add_documents(self, documents: List[str], embeddings: np.ndarray):
        if not self.collection:
            raise ValueError("ChromaDB collection is not initialized. Please check the initialization.")
        print(f"Adding {len(documents)} documents to the collection '{self.collection_name}'...")
        try:
            ids = []
            metadatas = []
            document_text = []
            embeddings_list = []
            
            for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                #Generate unique id
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(doc_id)
                
                #metadata prepare
                metadata = dict(doc.metadata)
                metadata['doc_index'] = i
                metadata['content_length'] = len(doc.page_content)
                metadatas.append(metadata)

                document_text.append(doc.page_content)
                embeddings_list.append(emb.tolist())
                
            try:
                self.collection.add(
                    ids=ids,
                    metadatas=metadatas,
                    documents=document_text,
                    embeddings=embeddings_list
                )
                print(f"Successfully added {len(documents)} documents to the collection '{self.collection_name}'.")
            except Exception as e:
                print(f"Error adding documents to the collection: {e}")
                
        except Exception as e:
            print(f"Error preparing documents for addition: {e}")
            
vector_store_manager = VectorStore()