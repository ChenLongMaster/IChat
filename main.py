from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import zipfile
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv

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


    @app.post("/bot")
    async def get_bot_response(req: PromptRequest):
        tenant_storage, data_dir = get_storage_paths(req.tenant_id)

        if not os.path.exists(os.path.join(tenant_storage, "docstore.json")):
            return {"error": f"No data found for tenant '{req.tenant_id}'"}

        prompt_path = os.path.join(data_dir, "Prompt", f"{req.tenant_id}_prompt.txt")
        print(f"🔍 Looking for prompt file at: {prompt_path}")

        system_prompt = DEFAULT_PROMPT
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
            print(f"📄 Loaded system prompt:\n{system_prompt}")
        else:
            print(f"⚠️ No custom prompt file found. Using default prompt.")

        storage_context = StorageContext.from_defaults(persist_dir=tenant_storage)
        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever(similarity_top_k=3)
        memory = ChatMemoryBuffer.from_defaults(token_limit=1000)

        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            memory=memory
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=req.user_prompt)
        ]

        response = llm.chat(messages)
        return {"response": response.message.content}

    @app.post("/reset")
    async def reset_memory():
        global memory
        memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
        return {"message": "Chat memory cleared."}

    @app.post("/upload-data/")
    async def upload_data(tenant_name: str = Form(...), file: UploadFile = File(...)):
        tenant_data_dir = f"data/{tenant_name}"
        tenant_storage_dir = f"storage/{tenant_name}"
        os.makedirs(tenant_data_dir, exist_ok=True)

        zip_path = os.path.join(tenant_data_dir, "upload.zip")
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tenant_data_dir)
        os.remove(zip_path)

        # 🛠️ Manually filter input files
        input_files = []
        for root, _, files in os.walk(tenant_data_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if exclude_prompt_folder(full_path):
                    input_files.append(full_path)

        documents = SimpleDirectoryReader(input_files=input_files).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(tenant_storage_dir)

        return {"message": f"Data for tenant '{tenant_name}' uploaded and indexed successfully."}

    return app
    

# === Run App ===
app = main_app()
