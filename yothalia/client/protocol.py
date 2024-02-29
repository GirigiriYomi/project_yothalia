import json 
import requests
from pydantic import BaseModel
import json

import edge_tts # to be deleted

# This is a model use to pass msg through FastAPI
class TextModel(BaseModel):
    text: str
    
class VoiceModel(BaseModel):
    voice: bytes 


class LlmProtocol():
    ip_config = json.load(open('./config.json','r'))['protocol_config'] # trigger problem if run this file as script, not module (depend on pwd)
    port = ip_config['port']
    ip = ip_config['ip']
    address = 'http://'+ip+':'+port
    
    @staticmethod
    async def send_text_to_asr(input_voice):
        # user text only, no system instruction
        url = LlmProtocol.address+'/asr_generate'
        headers = {'Content-Type':'application/json'}
        data = {'voice':input_voice}

        response = requests.post(url,headers=headers,data=json.dumps(data))
        response = response.json()['response']
        return response
    
    @staticmethod
    async def send_text_to_llm(input_text):
        # user text only, no system instruction
        url = LlmProtocol.address+'/llm_generate'
        headers = {'Content-Type':'application/json'}
        data = {'text':input_text}

        response = requests.post(url,headers=headers,data=json.dumps(data))
        response = response.json()['response']
        return response

    @staticmethod
    async def send_text_to_tts(input_text):
        # responde or any msg need to be voiced
        url = LlmProtocol.address+'/tts_generate'
        headers = {'Content-Type':'application/json'}
        data = {'text':input_text}

        response = requests.post(url,headers=headers,data=json.dumps(data))
        response = response.json()['response']
        return response
    
    @staticmethod
    async def send_text_to_edge(TEXT, voice = 'zh-CN-XiaoxiaoNeural', rate = '-8%', volume = '+0%'):
        response = b''
        communicate = edge_tts.Communicate(TEXT, voice, rate = rate, volume=volume)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                response+=(chunk["data"])
        
        return response






