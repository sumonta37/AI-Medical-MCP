import os

from dotenv import load_dotenv
from ollama import Client


# Load variables from .env
load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")


# Validate API key
if not api_key:
    raise ValueError(
        "OLLAMA_API_KEY is missing. "
        "Please check your .env file."
    )


# Create Ollama Cloud client
client = Client(
    host="https://ollama.com",
    headers={
        "Authorization": f"Bearer {api_key}"
    }
)


# Send a simple test
response = client.chat(
    model="gpt-oss:120b",
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in one sentence."
        }
    ]
)


print(response["message"]["content"])