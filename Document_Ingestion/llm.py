import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from retrival.rag_retrival import rag_retrieval_manager

load_dotenv()


openai_api_key = os.getenv("API_KEY")

llm = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1500,
)

def rag_simple(query, retrieval, llm, top_k=5, threshold=0.0):
    # Retrieve relevant documents using RAG
    retrieved_docs = retrieval.retrieve_documents(query, top_k=top_k, threshold=threshold)
    
    if not retrieved_docs:
        return "No relevant documents found."
    
    # Combine the content of the retrieved documents
    combined_content = "\n\n".join([doc["content"] for doc in retrieved_docs])
    
    # Create a prompt for the LLM
    prompt = f"Answer the following question based on the provided documents:\n\nQuestion: {query}\n\nDocuments:\n{combined_content}\n\nAnswer:"
    
    # Generate a response using the LLM
    response = llm.invoke(prompt)
    
    return response

response = rag_simple(
    "How many paid vacation days do full-time employees receive per year?",
    rag_retrieval_manager,
    llm,
    top_k=5,
    threshold=0.0,
)
print(response.content)