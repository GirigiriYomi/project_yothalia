import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain
from langchain.prompts import PromptTemplate

# from prompt_template import get_template

prompt_pure = """<s>[INTS]<<SYS>>
{instruction}<</SYS>>
{history}
{user}[/INST]
"""

prompt_intern ="""<|User|>:{query}<eoh>\n<|Bot|>:"""

class Baichuan2LLM:
    def __init__(self):
        print('Initializing model...')
        self.model = AutoModelForCausalLM.from_pretrained("D:/Projects/project_yothalia/yothalia/server/model_weights/internlm/internlm-chat-7b-finetune",
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("D:/Projects/project_yothalia/yothalia/server/model_weights/internlm/internlm-chat-7b-finetune",
                                                       trust_remote_code=True)
        

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.instruction = '你是一个可爱美少女'
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
        
        text = "User: "+text+'\n'
        response = self.llm_chain.run(instruction=self.instruction,history=self.history,user=text)
        self.history += text
        self.history += "Assistant: "+response+'\n'

        return response


class InterLM:
    def __init__(self):
        print('Initializing model...')
        self.model = AutoModelForCausalLM.from_pretrained("D:/Projects/project_yothalia/yothalia/server/model_weights/internlm/internlm-chat-7b-finetune",
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("D:/Projects/project_yothalia/yothalia/server/model_weights/internlm/internlm-chat-7b-finetune",
                                                       trust_remote_code=True)
        

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.instruction = '你是一个可爱美少女'
        self.history = ''
        self.llm_chain = self._pipeline_gen()
        print('Done initializing')
    
    def _pipeline_gen(self):
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=100, device=0)
        hf = HuggingFacePipeline(pipeline=pipe)
        prompt = PromptTemplate(template=prompt_intern, input_variables=['instruction','history','user'])
        return LLMChain(prompt=prompt, llm=hf)

    def predict(self,text):
        print(self.history)
        
        text = "User: "+text+'\n'
        response = self.llm_chain.run(instruction=self.instruction,history=self.history,user=text)
        self.history += text
        self.history += "Assistant: "+response+'\n'

        return response











