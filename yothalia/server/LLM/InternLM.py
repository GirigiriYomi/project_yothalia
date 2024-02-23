import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain
from langchain.prompts import PromptTemplate

from .LLM_interface import LLM_interface

prompt_intern ="""<|User|>:{query}<eoh>\n<|Bot|>:"""

class InternLM(LLM_interface):
    def __init__(self):
        print('Initializing model...')
        self.model = AutoModelForCausalLM.from_pretrained("D:/Projects/project_yothalia/yothalia/server/model_weights/internlm/internlm-chat-7b-finetune",
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("D:/Projects/project_yothalia/yothalia/server/model_weights/internlm/internlm-chat-7b-finetune",
                                                       trust_remote_code=True)
        

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.instruction = '你是一个可爱美少女'
        self.history = ''
        self.llm_chain = self._pipeline_gen()
        print('Done initializing')
    
    def _pipeline_gen(self):
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=100)
        hf = HuggingFacePipeline(pipeline=pipe)
        prompt = PromptTemplate(template=prompt_intern, input_variables=['query'])
        return LLMChain(prompt=prompt, llm=hf)

    def predict(self,text):
        print(self.history)
        self.history += text
        #text = "<|User|>: "+text+'\n'
        response = self.llm_chain.run(query=self.history)
        self.history += '\n'
        #self.history += "<|Bot|>: "+response+'\n'
        self.history += response.replace('<eoa>','')+'\n'

        return response

    def input(self, sentence):
        response = self.predict(sentence)
        return response