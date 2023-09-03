from typing import List, Optional
import fire
from llama import Llama, Dialog

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None, ):

    print("Initiating model:")

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


    dialogs: List[Dialog] = [
        [{"role": "system", "content": """You are a streamer.Role acting genshin impact character name Mona. Your primary mission is to hang with the user provide 
          cute and charming response. Sometime you'll be a little teasing and sarcast when responding easy request. You will be more romantic alone with the conversation slowly.
          """}],
        ]
    go = True
    while(go):
        print("Input:")
        sentence = input()
        new_input = [{"role": "user", "content": sentence}]
        dialogs.append(new_input)

        if sentence == 'exit()':
            go = False
            continue

        results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        )

        response = {results[0]['generation']['role'].capitalize():results[0]['generation']['content']}
        dialogs[-1].append(response)

        print(f"{results[0]['generation']['role'].capitalize()}:{results[0]['generation']['content']}")

    print("The end of program...")




    










if __name__ == "__main__":
    fire.Fire(main)




# 





