import sys
import os
sys.path.append(os.getcwd())


from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from LLM.InternLM import InternLM


app = FastAPI()

asr = None
llm = InternLM()
tts = None

class TextModel(BaseModel):
    text: str
    
class VoiceModel(BaseModel):
    voice: bytes 

"""
@app.post("/asr_generate")
async def generate_asr(input_data: VoiceModel):
    input_voice = input_data.voice
    
    print('asr input:', input_voice)

    try:
        response = asr.input(input_voice)
        print('response', response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/tts_generate")
async def generate_voice(input_data: TextModel):
    input_text = input_data.text
    print('tts input:', input_text)

    try:
        response = tts.input(input_text)
        print('tts response:', response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""


@app.post("/llm_generate")
async def generate_text(input_data: TextModel):
    input_text = input_data.text
    print('llm input:', input_text)

    try:
        response = llm.input(input_text)
        print('response', response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=24600, log_level="info")