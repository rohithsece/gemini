import json
import requests
from pathlib import Path

# Adjust these values as needed
url = "http://127.0.0.1:8000/run"
payload = {
    "query": "What is Retrieval Augmented Generation?",
    "docs_dir": str(Path(r"c:/Users/rocks/Desktop/RAG model").resolve()),
    "retriever_mode": "bm25",
    "model": "groq-llama3-8b",
    "api_key": "YOUR_GROQ_API_KEY",
    "model_info": {"vision": False, "function_calling": True, "json_output": True, "family": "unknown", "structured_output": True},
    "chat_messages": []
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print("Status:", response.status_code)
print("Response:", response.text)
