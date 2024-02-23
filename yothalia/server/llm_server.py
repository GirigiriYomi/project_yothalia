import sys
import os
sys.path.append(os.getcwd())


from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from ..client.protocol import TextModel, VoiceModel

from llms import Baichuan2LLM
from LLM.InternLM import InternLM


app = FastAPI()


llm = InternLM()
tts = None

@app.post("/llm_generate")
async def generate_text(input_data: TextModel):
    input_text = input_data.text
    print('input data:', input_text)

    
    try:
        response = llm.input(input_text)
        print('response', response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/tts_generate")
async def generate_voice(input_data: TextModel):
    input_text = input_data.text
    print('input data:', input_text)

    try:
        response = tts.input(input_text)
        print('response', response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_level="info")