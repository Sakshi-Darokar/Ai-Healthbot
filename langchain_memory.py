# langchain_memory.py
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

class ChatMemoryWrapper:
    def __init__(self, buffer):
        self.buffer = buffer

    def add_user_message(self, msg):
        self.buffer.append({"type": "human", "data": msg})

    def add_ai_message(self, msg):
        self.buffer.append({"type": "ai", "data": msg})

def get_memory_llm():
    llm = ChatOpenAI(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.5
    )

    memory = ConversationBufferMemory(return_messages=True)
    # Attach a wrapper with the required methods
    memory.chat_memory = ChatMemoryWrapper(memory.buffer)
    return llm, memory
