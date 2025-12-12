import os
from io import BytesIO
from typing import Dict, List
from dotenv import dotenv_values
from ollama import ChatResponse, chat


class OllamaUtils:

    def __init__(self):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env")
        self.model = dotenv_values(env_path)["MODEL"]
        self.ollama_url = dotenv_values(env_path)["OLLAMA_URL"]

    def ollama_init(self):
        os.system(f"ollama pull {self.model}")

    def interact(self, messages: List[Dict]):
        response: ChatResponse = chat(model=self.model, messages=messages)
        return response["message"]["content"]
