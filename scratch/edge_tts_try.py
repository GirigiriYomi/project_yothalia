import edge_tts
import asyncio
from pydub import AudioSegment
from pydub.playback import play
import io


TEXT = "我是一个蠢货ai"

voice = 'zh-CN-XiaoxiaoNeural'
output = '4.mp3'
rate = '-4%'
volume = '+0%'
async def my_function():
    tts = edge_tts.Communicate(text = TEXT,voice = voice,rate = rate,volume=volume)
    
    await tts.save(output)
    sound = AudioSegment.from_mp3(output)
    play(sound)


async def amain() -> bytes:
    result = b''
    communicate = edge_tts.Communicate(TEXT, voice)
    async for chunk in communicate.stream():
        print(chunk)
        if chunk["type"] == "audio":
            result+=(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            print(f"WordBoundary: {chunk}")
    sound = AudioSegment.from_mp3(io.BytesIO(result))
    play(sound)  
    

    
    
if __name__ == '__main__':
    #asyncio.run(my_function())
    asyncio.run(amain())
    #sound = AudioSegment.from_wav('D:/Projects/GPT-SoVITS-beta0217/raw/raina/raina_raw_1_0222.wav')
    #sound = AudioSegment.from_mp3('4.wav')
    #play(sound)
    
    
    
    
    