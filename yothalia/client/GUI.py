import gradio as gr
from ClientManager import Client
import asyncio

"""
control_dict = {
        'stop':False,
        'read_danmu':True,
        'accept_danmu':True
        }

def dict_set_default():
    global control_dict
    control_dict = {
        'stop':False,
        'read_danmu':True,
        'accept_danmu':True
        }
    
"""


def switch_on(): 
    client.stop.clear()
    asyncio.run(client.main())
    print('skip?')
        
def switch_off():
    print("enter")
    if not client.stop.is_set():
        client.stop.set()
        print("all set")
    else:
        print("已经结束了！不可以再结束了！！")


def listen_danmu(rec_danmu):
    print(rec_danmu)
    if rec_danmu:
        if not client.danmuManager:
            client.create_danmu_process()
            return gr.update(interactive=True,value=True)
    else:
        if client.danmuManager:
            client.destroy_danmu_process()
            return gr.update(interactive=False,value=False)
    

"""
async def main():
    
    with gr.Blocks() as demo:
        with gr.Row():
            start = gr.Button("开启客户端")
            end = gr.Button("结束客户端")
            start.click(fn=switch_on)
            end.click(fn=switch_off)
            
            with gr.Column():
                rec_danmu = gr.Checkbox(label="接收弹幕？",info="接收或屏蔽弹幕",value=True,interactive=True)
                d_reader = gr.Checkbox(label="朗读弹幕？",info="弹幕朗读",value=True,interactive=True)
                
                rec_danmu.change(fn=listen_danmu,inputs=rec_danmu,outputs=d_reader)
                
    client_task = asyncio.create_task(client.main())
    
    demo.launch()
"""
with gr.Blocks() as demo:
        with gr.Row():
            start = gr.Button("开启客户端")
            end = gr.Button("结束客户端")
            start.click(fn=switch_on)
            end.click(fn=switch_off)
            
            with gr.Column():
                rec_danmu = gr.Checkbox(label="接收弹幕？",info="接收或屏蔽弹幕",value=True,interactive=True)
                d_reader = gr.Checkbox(label="朗读弹幕？",info="弹幕朗读",value=True,interactive=True)
                
                rec_danmu.change(fn=listen_danmu,inputs=rec_danmu,outputs=d_reader)


if __name__ == '__main__':
    client = Client()
    demo.launch()
    
    