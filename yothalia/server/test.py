import requests
import json

def send_text(input_text):
    url = 'http://localhost:5000/generate'
    headers = {'Content-Type':'application/json'}
    data = {'text':input_text}

    response = requests.post(url,headers=headers,json=data)
    return response.json()



if __name__=='__main__':
    stop = False
    while not stop:
        input_text = input('Say something:')
        if input_text=='EXIT':
            stop = True
            continue
        print(send_text(input_text))












