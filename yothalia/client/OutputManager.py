import asyncio
from protocol import LlmProtocol

from pydub import AudioSegment
from pydub.playback import play
import io

import time

def run_output(queue):
    print("音频输出模块已就绪！")
                
    while True:
        asyncio.run(edge_play_sound(queue))
        time.sleep(0.5)

async def edge_play_sound(queue):
    if not queue.empty():
        text = queue.get()
        danmu_sound = await LlmProtocol.send_text_to_edge(text)
        sound = AudioSegment.from_mp3(io.BytesIO(danmu_sound))
        play(sound)






