import json 
import requests
from pydantic import BaseModel

# This is a model use to pass msg through FastAPI
class LLMModel(BaseModel):
    text: str

class LlmProtocol():

    @staticmethod
    def send_text_to_llm(input_text):
        # user text only, no system instruction
        url = 'http://localhost:8000/generate'
        headers = {'Content-Type':'application/json'}
        data = {'text':input_text}

        response = requests.post(url,headers=headers,data=json.dumps(data))
        response = response.json()['response']
        return response

    @staticmethod
    def send_text_to_tts(input_text):
        pass






