import os
from io import BytesIO
from typing import Dict, List
from dotenv import dotenv_values
from ollama import ChatResponse, chat


class OllamaUtils:

    def __init__(self):
        model = dotenv_values("../.env")["MODEL"]
        ollama_url = dotenv_values("../.env")["OLLAMA_URL"]

    def ollama_init(self):
        os.system(f"ollama pull {self.model}")

    def interact(self, messages: Dict):
        response: ChatResponse = chat(model=self.model, messages=messages)
        return response["message"]["content"]
