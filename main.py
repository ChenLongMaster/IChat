from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
import json
import traceback
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    SimpleDirectoryReader, StorageContext, VectorStoreIndex
)
from llama_index.core.settings import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.readers.web import BeautifulSoupWebReader

# === Load ENV ===
load_dotenv()

# === Constants ===
USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
DEFAULT_PROMPT = "You are a helpful assistant."
UPLOAD_FOLDER = r"D:\\IChat.Sources\\Upload"
PROMPT_FOLDER = r"D:\\IChat.Sources\\Instruction"
GENERIC_SETTING_FOLDER = r"D:\\IChat.Sources\\GeneralSetting"

# === Models ===
class PromptRequest(BaseModel):
    tenant_id: str
    user_prompt: str

class CrawlRequest(BaseModel):
    tenant_id: str
    vector_id: str
    url: str

class TrainRequest(BaseModel):
    tenant_id: str

# === Main app function ===
def main_app():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    @app.post("/bot")
    async def get_bot_response(req: PromptRequest):
        prompt_folder = os.path.join(PROMPT_FOLDER, req.tenant_id)
        system_prompt = DEFAULT_PROMPT
        temperature = 0.5
        max_tokens = 500

        if os.path.exists(prompt_folder):
            prompt_files = sorted(os.path.join(dp, f) for dp, _, filenames in os.walk(prompt_folder) for f in filenames if f.endswith(".txt"))
            prompt_chunks = [open(path, "r", encoding="utf-8").read() for path in prompt_files]
            system_prompt = "\n\n".join(prompt_chunks)

        setting_path = os.path.join(GENERIC_SETTING_FOLDER, req.tenant_id)
        if os.path.exists(setting_path):
            for dp, _, files in os.walk(setting_path):
                for file in files:
                    if not file.endswith(".txt"): continue
                    content = open(os.path.join(dp, file), "r", encoding="utf-8-sig").read().strip()
                    if content:
                        try:
                            config = json.loads(content)
                            temperature = config.get("Temparature", temperature)
                            max_tokens = config.get("MaxTokens", max_tokens)
                        except json.JSONDecodeError as e:
                            print(f"❌ JSON decode error in {file}: {e}")

        context = ""
        try:
            qdrant_client = QdrantClient(host="localhost", port=6333)
            collections = qdrant_client.get_collections().collections
            if req.tenant_id in [col.name for col in collections]:
                vector_store = QdrantVectorStore(client=qdrant_client, collection_name=req.tenant_id)
                index = VectorStoreIndex.from_vector_store(vector_store, storage_context=StorageContext.from_defaults(vector_store=vector_store))
                retriever = index.as_retriever(similarity_top_k=3)
                nodes = retriever.retrieve(req.user_prompt)
                context = "\n\n".join([n.get_content() for n in nodes])
        except Exception as e:
            print(f"❌ Qdrant access failed: {str(e)}")

        user_prompt = f"{req.user_prompt}\n\nContext:\n{context}"
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        payload = {
            "model": "llama3",
            "prompt": full_prompt,
            "stream": False
        }

        async with httpx.AsyncClient(timeout=360) as client:
            try:
                r = await client.post("https://ollama.s3corp.com.vn/api/generate", json=payload)
                r.raise_for_status()
                data = r.json()
                return {"response": data.get("response", str(data))}
            except Exception:
                print(traceback.format_exc())
                raise HTTPException(status_code=500, detail="LLM call failed. See logs for details.")

    @app.post("/train-rag")
    async def train_rag(train_request: TrainRequest):
        tenant_id = train_request.tenant_id
        tenant_data_dir = os.path.join(UPLOAD_FOLDER, tenant_id)
        input_files = [os.path.join(dp, f) for dp, _, files in os.walk(tenant_data_dir) for f in files if f.lower() != "config.txt"]
        if not input_files:
            raise HTTPException(status_code=400, detail="No valid files found to process.")

        documents = SimpleDirectoryReader(input_files=input_files).load_data()
        vector_store = QdrantVectorStore(client=QdrantClient(host="localhost", port=6333), collection_name=tenant_id)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        return {"message": f"RAG model for tenant '{tenant_id}' trained successfully."}

    @app.post("/crawl-job")
    async def crawl_job(crawl_request: CrawlRequest):
        url = crawl_request.url
        tenant_id = crawl_request.tenant_id
        vector_id = crawl_request.vector_id

        try:
            domain = urlparse(url).netloc.replace("www.", "")
            loader = BeautifulSoupWebReader()
            documents = loader.load_data(urls=[url])

            if not documents:
                raise HTTPException(status_code=404, detail="No content found on the page.")

            crawl_dir = os.path.join(UPLOAD_FOLDER, tenant_id, "crawl")
            os.makedirs(crawl_dir, exist_ok=True)

            for doc in documents:
                doc.metadata = {"tenant_id": tenant_id, "source_type": "web_crawl", "source_origin": url, "vector_id": vector_id}
                file_path = os.path.join(crawl_dir, f"{domain}_{vector_id}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"URL: {url}\n\n" + "\n".join(line.strip() for line in doc.text.splitlines() if line.strip()))

            return {"message": "Content fetched and saved.", "url": url, "vector_id": vector_id, "document_count": len(documents)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch and save URL: {str(e)}")

    return app

# === Run App ===
app = main_app()
