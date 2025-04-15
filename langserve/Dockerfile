FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y g++ build-essential
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install langchain-ollama langchain-chroma
RUN pip install chromadb==0.3.21
RUN pip install pydantic-settings
COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
