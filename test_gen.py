import os
from typing import List, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.3, max_tokens=1024, api_key=key)

messages = [
    SystemMessage(content="You are an expert OpenSCAD programmer. Generate a sphere."),
    HumanMessage(content="Generate a sphere.")
]

try:
    resp = llm.invoke(messages)
    print("Content:")
    print(repr(resp.content))
except Exception as e:
    print("Error:", e)
