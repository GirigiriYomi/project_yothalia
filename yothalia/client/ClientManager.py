import asyncio
import DanmuManager, OutputManager
from multiprocessing import Queue, Process
from protocol import LlmProtocol



import time

class Client():
    
    def __init__(self, control_dict={}):
        #super(Client,self).__init__()
        """
        control dict structure:
            {
                'stop':True,
                'read_danmu':True,
                'accept_danmu':True
            }
        """
        self.stop = asyncio.Event()
        self.read_danmu = True # TODO GUI control
        self.accept_danu = True
        
        self.danmu_queue = Queue()
        #self.input_queue = Queue()
        self.output_queue = Queue()
        
        self.danmuManager = None
        self.inputManager = None
        self.outputManager = None
        
        pass

    def create_danmu_process(self):
        self.danmuManager = Process(target=DanmuManager.run_bilibili, args=(self.danmu_queue,))
    def create_input_process(self):
        self.inputManager = None
    def create_output_process(self):
        self.outputManager = Process(target=OutputManager.run_output,args=(self.output_queue,))

    def destroy_danmu_process(self):
        self.danmuManager.terminate()
        self.danmuManager.join()
    def destroy_input_process(self):
        pass
    def destroy_output_process(self):
        self.outputManager.terminate()
        self.outputManager.join()
    
    async def main(self):
        print("客户端启动中。。")
        self.create_danmu_process()
        self.create_input_process()
        self.create_output_process()
        self.danmuManager.start()
        #self.inputManager.start()
        self.outputManager.start()
        
        while not self.stop.is_set():
            #print(Client.danmu_queue.empty())
            #print('recieving...')
            
            asyncio.gather(
                self.receive_danmu())
            
            #asyncio.run(self.receive_danmu())
            #asyncio.run(self.receive_input())
            
 
            time.sleep(2)
            
        print("正在结束...")
        if self.danmuManager and self.outputManager:
            self.destroy_danmu_process()
            self.destroy_input_process()
            self.destroy_output_process()
        print("客户端进程结束")
        
    async def receive_danmu(self):
        if not self.danmu_queue.empty():
            if not self.accept_danu:
                return
            danmu = self.danmu_queue.get()
            print(danmu) 
            if self.read_danmu:
                self.output_queue.put(danmu[0]+'说'+danmu[1])
            response = await LlmProtocol.send_text_to_llm(danmu[0]+'说'+danmu[1])
            print(response)
            
            
                
    
    """
    async def receive_edge(self, text):
        danmu_sound = await LlmProtocol.send_text_to_edge(text)
        sound = AudioSegment.from_mp3(danmu_sound)
        play(sound)
    """      
    
        
        
            
"""

    async def receive_voice(self, text):
            voice_response = await LlmProtocol.send_text_to_tts(text)
            Client.danmuManager.output_queue.put(voice_response)
        

    async def receive_input(self):
        if not Client.input_queue.empty():
            input = Client.danmuManager.danmu_queue.get()
            if input[0] == 'text':
                llm_response = await LlmProtocol.send_text_to_llm('JD蛋蛋'+': '+input[1])
                print(llm_response)
            else:
                asr_response = await LlmProtocol.send_text_to_asr(input[1])
                print(asr_response)
                llm_response = await LlmProtocol.send_text_to_llm('JD蛋蛋'+': '+asr_response)
                print(llm_response)
            
            tts_response = await self.receive_voice(llm_response)
            Client.output_queue.put(tts_response)
"""
    








if __name__ == "__main__":
    client = Client()
    client.main()
    
    #弹幕抗压测试 测试服务端模型是否能async -> Fine


