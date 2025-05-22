from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import zipfile
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from requests.exceptions import HTTPError
import requests
from fastapi.responses import JSONResponse
import traceback
import shutil

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

# === Load ENV ===
load_dotenv()

# === Constants ===
MODEL_PATH = os.getenv("MODEL_PATH")
USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
DEFAULT_PROMPT = "You are a helpful assistant."

# Custom file filter to exclude files from the 'Prompt' directory
def exclude_prompt_folder(filepath: str) -> bool:
    return "Prompt" not in os.path.normpath(filepath)

# === Models ===
class PromptRequest(BaseModel):
    tenant_id: str
    user_prompt: str

# === Helpers ===
def get_model_kwargs():
    kwargs = {"n_threads": 8}
    kwargs["n_gpu_layers"] = -1 if USE_GPU else 0
    print("🚀 Using GPU" if USE_GPU else "💻 Using CPU")
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
        f"data/{tenant_id}",
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


    from huggingface_hub import InferenceClient

    @app.post("/bot")
    async def get_bot_response(req: PromptRequest):
        _, data_dir = get_storage_paths(req.tenant_id)

        # Load system prompt
        prompt_path = os.path.join(data_dir, "Prompt", f"{req.tenant_id}_prompt.txt")
        print(f"🔍 Looking for prompt file at: {prompt_path}")

        system_prompt = DEFAULT_PROMPT
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
            print(f"📄 Loaded system prompt:\n{system_prompt}")
        else:
            print(f"⚠️ No custom prompt file found. Using default prompt.")

        # 🔍 Check if Qdrant collection exists
        try:
            qdrant_client = QdrantClient(host="localhost", port=6333)
            collections = qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]

            if req.tenant_id not in collection_names:
                return {"error": f"❌ No vector index found for tenant '{req.tenant_id}'"}

            # RAG: Load index from Qdrant
            vector_store = QdrantVectorStore(client=qdrant_client, collection_name=req.tenant_id)
            index = VectorStoreIndex.from_vector_store(vector_store)
            retriever = index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(req.user_prompt)
            context = "\n\n".join([n.get_content() for n in nodes])

        except Exception as e:
            return {"error": f"🚫 Failed to access vector DB for tenant '{req.tenant_id}': {str(e)}"}

        # 👤 Final constructed prompt with context
        user_prompt = f"{req.user_prompt}\n\nContext:\n{context}"

        # 🤖 Call Hugging Face Inference API
        from huggingface_hub import InferenceClient
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
                return {"error": "🚦 Hugging Face rate limit reached. Please try again later."}
            elif status == 401:
                return {"error": "🔐 Invalid or missing Hugging Face API token."}
            elif status == 403:
                return {"error": "⛔ Access to this model is restricted or you’ve hit your token quota."}
            else:
                return {"error": f"❌ Hugging Face API error: {status} - {text}"}

        except requests.exceptions.RequestException as e:
            return {"error": f"🌐 Network error while contacting Hugging Face: {str(e)}"}

        except Exception as e:
            return {"error": f"🔥 Unexpected error: {str(e)}"}


    @app.post("/reset")
    async def reset_memory():
        global memory
        memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
        return {"message": "Chat memory cleared."}

    

    @app.post("/upload-data/")
    async def upload_data(tenant_name: str = Form(...), file: UploadFile = File(...)):
        try:
            tenant_data_dir = f"data/{tenant_name}"
            os.makedirs(tenant_data_dir, exist_ok=True)

            zip_path = os.path.join(tenant_data_dir, "upload.zip")
            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tenant_data_dir)
            os.remove(zip_path)

            input_files = []
            for root, _, files in os.walk(tenant_data_dir):
                for f in files:
                    full_path = os.path.join(root, f)
                    if exclude_prompt_folder(full_path):
                        input_files.append(full_path)

            documents = SimpleDirectoryReader(input_files=input_files).load_data()

            # 🔍 Wrap this part to catch Qdrant issues
            try:
                client = QdrantClient(host="localhost", port=6333)
                vector_store = QdrantVectorStore(client=client, collection_name=tenant_name)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            except Exception as vector_err:
                return JSONResponse(status_code=500, content={"error": f"Qdrant error: {str(vector_err)}"})

            return {"message": f"Data for tenant '{tenant_name}' uploaded and indexed successfully."}

        except Exception as e:
            return JSONResponse(status_code=500, content={
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

            return {"message": "✅ All tenant data cleared from data/ and storage/."}

        except Exception as e:
            return {"error": f"❌ Failed to clear data: {str(e)}"}

    return app
    

# === Run App ===
app = main_app()
