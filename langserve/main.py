from fastapi import FastAPI
from langserve import add_routes
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma  # Updated import
from langchain_ollama import OllamaEmbeddings  # Updated import
from chromadb import Client
from chromadb.config import Settings  # Reintroduce Settings import
import os
import requests

app = FastAPI()

embedding = OllamaEmbeddings(model="mistral", base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

# Test connectivity to the Chroma server
chroma_host = os.environ.get("CHROMA_HOST", "chroma")
chroma_port = os.environ.get("CHROMA_PORT", 8000)
try:
    response = requests.get(f"http://{chroma_host}:{chroma_port}/api/v2/heartbeat")
    print("Chroma server connectivity test:", response.status_code, response.text)
except Exception as e:
    print("Failed to connect to Chroma server:", e)

# Initialize the Chroma client with default local API implementation
client_settings = Settings(
    chroma_server_host=chroma_host,  # Optional if using the default local API
    chroma_server_http_port=int(chroma_port)  # Optional if using the default local API
)

chroma_client = Client(settings=client_settings)

vectordb = Chroma(
    collection_name="my_docs",
    embedding_function=embedding,
    client=chroma_client
)

qa = RetrievalQA.from_chain_type(
    retriever=vectordb.as_retriever(),
    chain_type="stuff"
)

add_routes(app, qa, path="/qa")
