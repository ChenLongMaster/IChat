from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import zipfile
from fastapi import FastAPI, UploadFile, File, Form,HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from requests.exceptions import HTTPError
import requests
from fastapi.responses import JSONResponse
import traceback
import shutil
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

from llama_cpp import Llama
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.readers.web import BeautifulSoupWebReader

import uuid
# === Load ENV ===
load_dotenv()

# === Constants ===
MODEL_PATH = os.getenv("MODEL_PATH")
USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
DEFAULT_PROMPT = "You are a helpful assistant."
ROOT_FOLDER = r"D:\\IChat.Sources\\Upload"

# Custom file filter to exclude files from the 'Prompt' directory
def exclude_prompt_folder(filepath: str) -> bool:
    return "Prompt" not in os.path.normpath(filepath)

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
# === Helpers ===
def get_model_kwargs():
    kwargs = {"n_threads": 8}
    kwargs["n_gpu_layers"] = -1 if USE_GPU else 0
    print("Using GPU" if USE_GPU else "Using CPU")
    return kwargs

def load_system_prompt(data_dir: str, tenant_id: str) -> str:
    prompt_path = os.path.join(data_dir, f"{tenant_id}_prompt.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_PROMPT

def get_storage_paths(tenant_id: str):
    return (
        f"storage/{tenant_id}",
        f"D:\\IChat.Sources\\Upload\\{tenant_id}"
    )

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

    llm = LlamaCPP(
        model_path=MODEL_PATH,
        temperature=0.5,
        max_new_tokens=250,
        context_window=2048,
        model_kwargs=get_model_kwargs(),
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True
    )

    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    @app.get("/clients")
    async def list_clients():
        client_root = "storage"
        if not os.path.exists(client_root):
            return {"clients": []}
        
        clients = [
            name for name in os.listdir(client_root)
            if os.path.isdir(os.path.join(client_root, name))
        ]
        return {"clients": clients}



    @app.post("/bot")
    async def get_bot_response(req: PromptRequest):
        prompt_path = os.path.join(ROOT_FOLDER, req.tenant_id, "Instruction", "instruction.txt")
        system_prompt = DEFAULT_PROMPT
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
            print(f"✅ System prompt loaded from {prompt_path}")
        else:
            print(f"⚠️ No custom prompt file found at {prompt_path}. Using default prompt.")

        # 🔍 Check if Qdrant collection exists
        try:
            qdrant_client = QdrantClient(host="localhost", port=6333)
            collections = qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]

            if req.tenant_id not in collection_names:
                raise HTTPException(status_code=500, detail=f"No vector index found for tenant '{req.tenant_id}'")

            vector_store = QdrantVectorStore(client=qdrant_client, collection_name=req.tenant_id)
            index = VectorStoreIndex.from_vector_store(vector_store)
            retriever = index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(req.user_prompt)
            context = "\n\n".join([n.get_content() for n in nodes])

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to access vector DB: {str(e)}")

        # 👤 Final constructed prompt with context
        user_prompt = f"{req.user_prompt}\n\nContext:\n{context}"

        print("✅ API key loaded:", os.getenv("HF_API_TOKEN"))
        client = InferenceClient(provider="cerebras", api_key=os.getenv("HF_API_TOKEN"))

        try:
            stream = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=500,
                top_p=0.7,
                stream=True
            )

            output = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if "content" in delta and delta["content"] is not None:
                        output += delta["content"]

            return {"response": output}

        except HTTPError as e:
            status = getattr(e.response, "status_code", "unknown")
            text = getattr(e.response, "text", str(e))
            if status == 429:
                raise HTTPException(status_code=429, detail=f"Hugging Face rate limit reached. Please try again later.")
            elif status == 401:
                raise HTTPException(status_code=401, detail=f"Invalid or missing Hugging Face API token.")
            elif status == 403:
                raise HTTPException(status_code=403, detail=f"Access to this model is restricted or you’ve hit your token quota.")
            else:
                raise HTTPException(status_code=500, detail=f"Inference API error: {str(e)}")

    @app.post("/reset")
    async def reset_memory():
        global memory
        memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
        return {"message": "Chat memory cleared."}

    import traceback

    @app.post("/train-rag")
    async def train_rag(train_request: TrainRequest):
        try:
            tenant_id = train_request.tenant_id
            tenant_data_dir = f"{ROOT_FOLDER}\\{tenant_id}"
            input_files = []

            for root, _, files in os.walk(tenant_data_dir):
                for f in files:
                    if f.lower() == "instruction.txt":
                        continue
                    full_path = os.path.join(root, f)
                    input_files.append(full_path)

            if not input_files:
                raise HTTPException(status_code=400, detail=f"No valid files found to process.")


            # Log each file path
            print(f"\n📂 Files being loaded for tenant '{tenant_id}':")
            for file_path in input_files:
                print(f" - {file_path}")

            documents = SimpleDirectoryReader(input_files=input_files).load_data()

            client = QdrantClient(host="localhost", port=6333)
            vector_store = QdrantVectorStore(client=client, collection_name=tenant_id)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            VectorStoreIndex.from_documents(documents, storage_context=storage_context)

            return {"message": f"RAG model for tenant '{tenant_id}' trained and updated successfully."}

        except Exception as e:
            print("❌ ERROR OCCURRED:")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail={
                "error": str(e),
                "trace": traceback.format_exc()
            })

    @app.post("/clear-all-data")
    async def clear_all_data():
        data_dir = "data"
        storage_dir = "storage"

        try:
            # Clear all tenant folders in data/
            if os.path.exists(data_dir):
                for tenant in os.listdir(data_dir):
                    tenant_path = os.path.join(data_dir, tenant)
                    if os.path.isdir(tenant_path):
                        shutil.rmtree(tenant_path)

            # Clear all tenant folders in storage/
            if os.path.exists(storage_dir):
                for tenant in os.listdir(storage_dir):
                    tenant_path = os.path.join(storage_dir, tenant)
                    if os.path.isdir(tenant_path):
                        shutil.rmtree(tenant_path)

            return {"message": "All tenant data cleared from data/ and storage/."}
        except Exception as e:
            return {"error": f"Failed to clear data: {str(e)}"}

    @app.post("/crawl-job")
    async def crawl_job(crawl_request: CrawlRequest):
        print(f"Request:{crawl_request}")
        url = crawl_request.url
        tenant_id = crawl_request.tenant_id
        vector_id = crawl_request.vector_id

        try:
            # Extract domain from the URL
            parsed_url = urlparse(url)
            website_domain = parsed_url.netloc.replace("www.", "")  # remove www if present

            # Initialize the BeautifulSoup-based web reader
            loader = BeautifulSoupWebReader()
            documents = loader.load_data(urls=[url])

            if not documents:
                raise HTTPException(status_code=404, detail="No content found on the page.")

            # Prepare save folder
            tenant_crawl_dir = f"{ROOT_FOLDER}\\{tenant_id}\\crawl"
            os.makedirs(tenant_crawl_dir, exist_ok=True)

            for idx, doc in enumerate(documents):
                # Add metadata
                doc.metadata = {
                    "tenant_id": tenant_id,
                    "source_type": "web_crawl",
                    "source_origin": url,
                    "vector_id": vector_id
                }

                # Build filename → domain + vector_id
                file_name = f"{website_domain}_{vector_id}.txt"
                file_path = os.path.join(tenant_crawl_dir, file_name)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(doc.text)

                print(f"Saved crawled content to {file_path}")

            return {
                "message": "Successfully fetched and saved content.",
                "url": url,
                "vector_id": vector_id,
                "document_count": len(documents)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch and save URL: {str(e)}")
    return app

# === Run App ===
app = main_app()
