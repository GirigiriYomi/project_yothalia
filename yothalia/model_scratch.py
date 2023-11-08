#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.generation.utils import GenerationConfig
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain
from langchain.prompts import PromptTemplate

from prompt_template import get_template

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# run this to download model weights
#from huggingface_hub import snapshot_download
#snapshot_download(repo_id="baichuan-inc/Baichuan2-7B-Chat-4bits", local_dir="./model_weights/baichuan-inc/Baichuan2-7B-Chat-4bits")
#snapshot_download(repo_id="baichuan-inc/Baichuan2-13B-Chat-4bits", local_dir="./model_weights/baichuan-inc/Baichuan2-13B-Chat-4bits")


# In[4]:


tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat-4bits", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./model_weights/baichuan-inc/Baichuan2-7B-Chat-4bits", load_in_4bit=True, 
                                             trust_remote_code=True)


# In[5]:


model = model.to('cuda')
tokenizer.bos_token = '[INST]'
tokenizer.eos_token = '[/INST]'
tokenizer.pad_token = tokenizer.eos_token


# In[6]:


pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, device=0
)
hf = HuggingFacePipeline(pipeline=pipe)
prompt = PromptTemplate(template=get_template(), input_variables=['instruction','history','user'])
llm_chain = LLMChain(prompt=prompt, llm=hf)


# In[7]:


llm_chain.run(instruction='你是一个热情可爱美少女',history='',user='你好')


# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


prompt


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat-4bits")
messages = []

messages.append({"role": "user", "content": "解释一下“温故而知新”"})
response = model.chat(tokenizer, messages)
print(response)


# In[ ]:


messages.append({"role": "assisstant", "content":response})
messages.append({"role": "user", "content": "请介绍一下自己"})
response = model.chat(tokenizer, messages)
print(response)

