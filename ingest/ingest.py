import os, json
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

embedding = OllamaEmbeddings(model="mistral", base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"))

docs = []
for fname in os.listdir("/data"):
    if fname.endswith(".json"):
        with open(f"/data/{fname}") as f:
            content = json.load(f)
            text = json.dumps(content, indent=2)
            docs.append(Document(page_content=text, metadata={"source": fname}))

Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="my_docs",
    client_settings={
        "chroma_api_impl": "rest",
        "chroma_server_host": os.environ.get("CHROMA_HOST", "chroma"),
        "chroma_server_http_port": int(os.environ.get("CHROMA_PORT", 8000))
    }
)
