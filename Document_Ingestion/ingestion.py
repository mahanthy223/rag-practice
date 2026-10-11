from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from chuncking import split_documents
from retrival.embedding import embedding_manager
from retrival.vectore_store import vector_store_manager
loader = TextLoader("../Documents/rag_practice_document.txt", encoding="utf8")
documents = loader.load()
# print(documents)

dir_loader = DirectoryLoader(
    "../Documents", 
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=False
    )

pdf_documents = dir_loader.load()

chunks = split_documents(pdf_documents)
print(f"Total chunks created: {len(chunks)}")

texts = [chunk.page_content for chunk in chunks]

embeddings = embedding_manager.generate_embeddings(texts)
vector_store_manager.add_documents(chunks, embeddings)
print("Document ingestion completed successfully.")
