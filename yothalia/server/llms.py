from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain
from langchain.prompts import PromptTemplate

# from prompt_template import get_template

prompt_pure = """<s>
你将要扮演一个{instruction}
{history}
{user}
"""

class Baichuan2LLM:
    def __init__(self):
        print('Initializing model...')
        self.model = AutoModelForCausalLM.from_pretrained("./model_weights/baichuan-inc/Baichuan2-7B-Chat-4bits", load_in_4bit=True, trust_remote_code=True, cache_dir='./model_weights/baichuan-inc/Baichuan2-7B-Chat-4bits').cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat-4bits", use_fast=False, trust_remote_code=True)
        

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.instruction = '可爱美少女'
        self.history = ''
        self.llm_chain = self._pipeline_gen()
        print('Done initializing')
    
    def _pipeline_gen(self):
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=100, device=0)
        hf = HuggingFacePipeline(pipeline=pipe)
        prompt = PromptTemplate(template=prompt_pure, input_variables=['instruction','history','user'])
        return LLMChain(prompt=prompt, llm=hf)

    def predict(self,text):
        print(self.history)
        
        response = self.llm_chain.run(instruction=self.instruction,history=self.history,user=text)
        self.history += "用户："+text+'\n'
        self.history += "少女: "+response+'\n'

        return response














