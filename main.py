from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core.settings import Settings
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH")

# Wrap llama-cpp inside LlamaIndex's LlamaCPP
llm = LlamaCPP(
    model_path=MODEL_PATH,
    temperature=0.7,
    max_new_tokens=300,
    context_window=2048,
    model_kwargs={"n_threads": 8},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True
)

# Apply to global settings (or use explicitly in each engine)
Settings.llm = llm

# Create memory and chat engine
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
chat_engine = SimpleChatEngine.from_defaults(
    memory=memory
)

class PromptRequest(BaseModel):
    user_prompt: str

@app.post("/bot")
async def get_bot_response(req: PromptRequest):
    response = chat_engine.chat(req.user_prompt)
    return {"response": response.response}


