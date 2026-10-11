from .vectore_store import VectorStore, vector_store_manager
from .embedding import Embedding, embedding_manager
from pprint import pprint
from typing import List, Dict


class RagRetrieval:
    def __init__(self, vector_store: VectorStore, embedding_model: Embedding):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        
    def retrieve_documents(self, query: str, top_k: int = 5, threshold: float = 0.0)->List[Dict[str, any]]:
        # Generate embedding for the query
        query_embedding = self.embedding_model.generate_embeddings([query])[0]
        print(f"Query embedding generated with shape: {query_embedding.shape}")
        # Retrieve documents from the vector store
        results = self.vector_store.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Filter results based on threshold
        filtered_results = []
        
        if not results["documents"] or not results["documents"][0]:
            print("No documents found in the collection. Run ingestion.py first.")
            return filtered_results

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]
        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            relevance_score = 1 / (1 + distance)
            
            if relevance_score >= threshold:
                filtered_results.append({
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                })

        print(f"Retrieved {len(filtered_results)} documents (after filtering)")
        pprint(filtered_results, sort_dicts=False)
        return filtered_results
    
rag_retrieval_manager = RagRetrieval(vector_store=vector_store_manager, embedding_model=embedding_manager)

rag_retrieval_manager.retrieve_documents("How many paid vacation days do full-time employees receive per year?")
        