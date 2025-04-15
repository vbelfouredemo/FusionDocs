import os
import json
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import chromadb

# Set up the embedding model
embedding = OllamaEmbeddings(
    model="mistral",
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
)

# Load documents from /data
docs = []
for fname in os.listdir("/data"):
    if fname.endswith(".json"):
        with open(f"/data/{fname}") as f:
            content = json.load(f)
            text = json.dumps(content, indent=2)
            docs.append(Document(page_content=text, metadata={"source": fname}))

# Connect to remote ChromaDB REST API
chroma_client = chromadb.HttpClient(
    host=os.environ.get("CHROMA_HOST", "chroma"),
    port=int(os.environ.get("CHROMA_PORT", 8000))
)

# Index documents
Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="my_docs",
    client=chroma_client
)
