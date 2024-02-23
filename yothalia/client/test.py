import requests
import json
from protocol import LlmProtocol

"""
def send_text(input_text):
    url = 'http://localhost:8000/generate'
    headers = {'Content-Type':'application/json'}
    data = {'text':input_text}

    response = requests.post(url,headers=headers,data=json.dumps(data))
    return response.json()
"""




if __name__=='__main__':
    stop = False
    while not stop:
        input_text = input('Say something:')
        if input_text=='EXIT':
            stop = True
            continue
        print(LlmProtocol.send_text_to_llm(input_text))












