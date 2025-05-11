from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from llama_cpp import Llama
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core.settings import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os, shutil, zipfile
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH")
use_gpu = os.getenv("USE_GPU", "False").lower() == "true"

# 🧠 Default system prompt
DEFAULT_PROMPT = "You are a helpful assistant."

# ⚙️ Inference settings
model_kwargs = {"n_threads": 8}
if use_gpu:
    model_kwargs["n_gpu_layers"] = -1
    print("🚀 Using GPU for inference")
else:
    print("💻 Using CPU for inference")

# 🧠 Initialize the LLM and set global settings
llm = LlamaCPP(
    model_path=MODEL_PATH,
    temperature=0.7,
    max_new_tokens=300,
    context_window=2048,
    model_kwargs=model_kwargs,
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True
)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 📦 Request model
class PromptRequest(BaseModel):
    tenant_id: str
    user_prompt: str

@app.post("/bot")
async def get_bot_response(req: PromptRequest):
    tenant_id = req.tenant_id
    tenant_storage = f"storage/{tenant_id}"
    data_dir = f"data/{tenant_id}"

    if not os.path.exists(os.path.join(tenant_storage, "docstore.json")):
        return {"error": f"No data found for tenant '{tenant_id}'"}

    # Load tenant-specific prompt
    prompt_path = os.path.join(data_dir, f"{tenant_id}_prompt.txt")
    
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        print(f"📝 Loaded system prompt for tenant '{tenant_id}'")
    else:
        system_prompt = DEFAULT_PROMPT
        print(f"⚠️ No prompt file found for tenant '{tenant_id}', using default")

    # Load tenant-specific index and engine
    storage_context = StorageContext.from_defaults(persist_dir=tenant_storage)
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=3)
    memory = ChatMemoryBuffer.from_defaults(token_limit=1000)

    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        system_prompt=system_prompt
    )

    response = chat_engine.chat(req.user_prompt)
    return {"response": response.response}


# 🔄 Memory reset (shared/global only)
@app.post("/reset")
async def reset_memory():
    global memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
    return {"message": "Chat memory cleared."}

# 📤 Upload ZIP of PDFs, TXTs, etc.
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

    # Load and index documents
    documents = SimpleDirectoryReader(tenant_data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(tenant_storage_dir)

    return {"message": f"Data for tenant '{tenant_name}' uploaded and indexed successfully."}
