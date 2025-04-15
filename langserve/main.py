from fastapi import FastAPI
from langserve import add_routes
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_community.embeddings import OllamaEmbeddings  # Updated import
from chromadb.config import Settings
import os

app = FastAPI()

embedding = OllamaEmbeddings(model="mistral", base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

print("Debugging Chroma Settings:")
print(Settings(
    chroma_api_impl="local",  # Current implementation
    chroma_server_host=os.environ.get("CHROMA_HOST", "localhost"),
    chroma_server_http_port=int(os.environ.get("CHROMA_PORT", 8000))
))

vectordb = Chroma(
    collection_name="my_docs",
    embedding_function=embedding,
    client_settings=Settings(
        chroma_api_impl="local",  # Switch to 'local' implementation
        chroma_server_host=os.environ.get("CHROMA_HOST", "localhost"),
        chroma_server_http_port=int(os.environ.get("CHROMA_PORT", 8000))
    )
)

qa = RetrievalQA.from_chain_type(
    retriever=vectordb.as_retriever(),
    chain_type="stuff"
)

add_routes(app, qa, path="/qa")
