#!/usr/bin/env python
# coding: utf-8

# In[1]:


from paddlespeech.server.engine.asr.python.asr_engine import ASREngine, PaddleASRConnectionHandler

from paddlespeech.cli.asr.infer import ASRExecutor
import pyaudio, wave
import numpy as np
import speech_recognition as sr
import io
import time


# In[2]:


import threading
import queue

class mic_thread(threading.Thread):
    threads = []
    id = 0
    thread_lock = threading.Lock()
    mic_queue = queue.Queue(10)
    
    def __init__(self, name, source):
        threading.Thread.__init__(self)
        self.name = name
        self.source = source
        self.stop = False
        
        self.id = mic_thread.id
        mic_thread.id = mic_thread.id+1
        
        mic_thread.threads.append(self)
        
    def run(self):
        with mic as source:
            r.adjust_for_ambient_noise(source) # This filters noise
            r.pause_threshold = 0.8
            print("Start talk..")
            while not self.stop:
                print("Listening..")
                audio = r.listen(source)
                print("Done lisening")
                wav_data = audio.get_wav_data(convert_rate = 16000)
                
                mic_thread.thread_lock.acquire()
                mic_thread.mic_queue.put(io.BytesIO(wav_data))
                mic_thread.thread_lock.release()
                
    def terminate(self):
        self.stop = True
                


# In[3]:


class crASR:

    def __init__(self):
        # 创建对象参数
        class _conf:
            def __init__(self, **kwargs):
                for key in kwargs:
                    setattr(self, key, kwargs[key])
        # 初始化ASR
        self.asr = ASREngine()
        # 以指定参数方式生成对象并初始化ASR
        self.asr.init(_conf(
            model='conformer_talcs',
            lang='zh_en',
            sample_rate=16000,
            cfg_path=None,  # 自定义配置文件路径(可选)
            ckpt_path=None,  # 自定义模型文件路径(可选)
            decode_method='attention_rescoring',
            force_yes=True,
            device=None))
        self.asr_handle = PaddleASRConnectionHandler(self.asr)

    def predict(self, wavData):
        """
        ASR预测
        :param wavData: 需要预测的音频，会根据传入类型自动识别：路径(str 相对|绝对)、音频流、内存BytesIO对象
        :return:
        """
        if type(wavData) == str:
            wavData = filesystem_get_contents(wavData, 'rb')
        elif type(wavData) == io.BytesIO:
            wavData = wavData.getvalue()

        start = time.time()
        self.asr_handle.run(wavData)
        text = self.asr_handle.output
        print("ASR预测消耗时间：%dms, 识别结果: %s" % (round((time.time() - start) * 1000), text))
        return text


# In[4]:


asr = crASR()
r = sr.Recognizer()
mic = sr.Microphone(device_index=1)


# In[5]:


sr.Microphone.list_microphone_names()


# In[6]:


microphone = mic_thread("local mic",mic)


# In[9]:


microphone.start()

flag = True
while flag:
    if not mic_thread.mic_queue.empty():
        print('拿取')
        mic_thread.thread_lock.acquire()
        msg = mic_thread.mic_queue.get(0)
        mic_thread.thread_lock.release()
        print('拿到')
        
        replyTxt = asr.predict(msg)
        
        # send to LLM afterward
        print(replyTxt)
        
        if (replyTxt == "停下"):
            flag = False
    
    time.sleep(0.1)


# In[13]:


microphone.terminate()


# In[16]:


mic_thread.mic_queue.qsize()


# In[17]:


mic_thread.threads


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


with mic as source:
    r.adjust_for_ambient_noise(source) # This filters noise
    r.pause_threshold = 1
    print("Start talk..")
    while True:
        print("Listening..")
        audio = r.listen(source)
        
    


# In[17]:


wav_data = audio.get_wav_data(convert_rate = 16000)
io_wav = io.BytesIO(wav_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


with open("audio_file.wav", "wb") as file:
    file.write(wav_data)


# In[ ]:





# In[13]:


asr = ASREngine()
asr.init({'model':'conformer_wenetspeech',
            'lang':'zh',
            'sample_rate':16000,
            'cfg_path':None,  # 自定义配置文件路径(可选)
            'ckpt_path':None,  # 自定义模型文件路径(可选)
            'decode_method':'attention_rescoring',
            'force_yes':True,
            'device':'cuda'})

