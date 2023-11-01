from multiprocessing import Process, Lock, Queue
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain
from langchain.prompts import PromptTemplate

import time
from prompt_template import get_template

import warnings
warnings.filterwarnings("ignore")

class LlmFrame:
    """
    A server like LLM process that keep listening to it's buffer and output to a output buffer
    """

    def __init__(self):
        print('Initializing Model...')
        start_time = time.time()
        self.process = Process(target=self.to_run, args=())
        self.message_buffer = Queue(maxsize=10)
        self.output_buffer = Queue(maxsize=10)
        
        
        self.model = AutoModelForCausalLM.from_pretrained("./model_weights/baichuan-inc/Baichuan2-7B-Chat-4bits", load_in_4bit=True, trust_remote_code=True, ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat-4bits", use_fast=False, trust_remote_code=True)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.instruction = '你是一个打哑谜的助手'
        self.history = ''
        self.llm_chain = self._pipeline_gen()

        print(f'Initializing Complete, {time.time()-start_time}sec')

    def _pipeline_gen(self):
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=100, device=0)
        hf = HuggingFacePipeline(pipeline=pipe)
        prompt = PromptTemplate(template=get_template(), input_variables=['instruction','history','user'])
        return LLMChain(prompt=prompt, llm=hf)


    def to_run(self):
        #listening buffer
        while True:

            if self.message_buffer.empty():
                time.sleep(0.1)
                continue
            
            #get response from model
            msg_input = self.__read_message()
        
            #process with llm
            llm_start = time.time()
            msg_input = self.infer(user=msg_input)
            print(f"Response process time {time.time()-llm_start:.2f}",flush=True)

            #output response
            self.__put_output(msg_input)

            #add history
            

    def start(self):
        self.process.start()
    def stop(self):
        self.process.terminate()

    def set_sys_prompt(self, system_prompt):
        #模型人物设定
        self.instruction = system_prompt

    # process safe put/get functions
    def put_message(self,text_msg):
        self.message_buffer.put(text_msg)
    
    def __read_message(self): # in class use only
        try:
            msg = self.message_buffer.get()
        except Queue.Empty:
            return ''
        return msg

    def __put_output(self,text_output): # in class use only
        self.output_buffer.put(text_output)
        
    def get_output(self):
        try:
            output = self.output_buffer.get()
        except Queue.Empty:
            return ''
        return output
    
    def infer(self, user=''):
        return self.llm_chain.run(instruction=self.instruction,history=self.history,user=user)
    




if __name__ == '__main__':
    """
    测试脚本 运行:
    python LlmFrame.py
    输入EXIT 或者按键 CTRL+C 结束程序
    """
    llm = LlmFrame()
    llm.start()

    stop = False
    while not stop:
        print('LLM process start, input EXIT or press CTRL+C to stop')
        msg = input('Say something:')
        if msg == 'EXIT':
            stop = True
            llm.stop()
            continue

        llm.put_message(msg)

        while True:
            if llm.output_buffer.empty():
                time.sleep(0.1)
                continue
            
            print(llm.get_output())
            break
        

        



