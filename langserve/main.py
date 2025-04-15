from fastapi import FastAPI
from langserve import add_routes
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import os

app = FastAPI()

embedding = OllamaEmbeddings(model="mistral", base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

vectordb = Chroma(
    collection_name="my_docs",
    embedding_function=embedding,
    client_settings={
        "chroma_api_impl": "rest",
        "chroma_server_host": os.environ.get("CHROMA_HOST", "localhost"),
        "chroma_server_http_port": int(os.environ.get("CHROMA_PORT", 8000))
    }
)

qa = RetrievalQA.from_chain_type(
    retriever=vectordb.as_retriever(),
    chain_type="stuff"
)

add_routes(app, qa, path="/qa")
