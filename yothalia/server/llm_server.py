from fastapi import FastAPI, HTTPException, Request
from llms import Baichuan2LLM
from pydantic import BaseModel
from protocol import LLMModel


app = FastAPI()

llm = Baichuan2LLM()


@app.post("/generate")
async def generate_text(input_data: LLMModel):
    input_text = input_data.text
    print('input data:', input_text)

    try:
        response = llm.predict(input_text)
        print('response', response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_level="info")