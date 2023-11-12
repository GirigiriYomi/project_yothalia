from flask import Flask, request, jsonify
from llms import Baichuan2LLM

app = Flask(__name__)

llm = Baichuan2LLM()

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    print('input data:',data)
    input_text = data.get('text', '')

    response = llm.predict(input_text)

    print('response',response)
    return jsonify({'response': response})

if __name__ == '__main__':
    

    app.run(debug=False)