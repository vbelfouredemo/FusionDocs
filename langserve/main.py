from fastapi import FastAPI
from langserve import add_routes
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma  # Updated import
from langchain_ollama import OllamaEmbeddings  # Updated import
from chromadb.config import Settings
import os

app = FastAPI()

embedding = OllamaEmbeddings(model="mistral", base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

vectordb = Chroma(
    collection_name="my_docs",
    embedding_function=embedding,
    client_settings=Settings(
        chroma_api_impl="rest",  # Switch to 'rest' implementation
        chroma_server_host=os.environ.get("CHROMA_HOST", "chroma"),  # Use the Chroma container hostname
        chroma_server_http_port=int(os.environ.get("CHROMA_PORT", 8000))  # Use the Chroma container port
    )
)

qa = RetrievalQA.from_chain_type(
    retriever=vectordb.as_retriever(),
    chain_type="stuff"
)

add_routes(app, qa, path="/qa")
