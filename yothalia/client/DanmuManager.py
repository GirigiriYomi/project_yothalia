from bilibili_api import live, sync, Credential
from multiprocessing import Queue
import json

# Read personal bilibili cookie info to connect room
danmu_info = json.load(open('./config.json','r'))['bilibili_danmu']

credential = Credential(
        sessdata = danmu_info['SESSDATA'],
        bili_jct = danmu_info['BILI_JCT'],
        buvid3 = danmu_info['BUVID3'],
        )
RoomID = danmu_info['ROOM_ID']


def run_bilibili(queue):
    room = live.LiveDanmaku(RoomID, credential=credential)
    
    @room.on('DANMU_MSG')
    async def on_danmaku(event):
        # 收到弹幕
        user_name = event['data']['info'][2][1]
        content = event['data']['info'][1]
        
        print(user_name+": "+content)
        queue.put((user_name,content))


    @room.on('SEND_GIFT')
    async def on_gift(event):
        # 收到礼物
        print(event)
        
    sync(room.connect())
    

    
"""
if __name__ == '__main__':
    queue = Queue()
    print('listening...')
    run(queue)

"""    


