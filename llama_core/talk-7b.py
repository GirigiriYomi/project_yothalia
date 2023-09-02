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

    go = True
    while(go):
        print("Input:")
        sentence = input()

        if sentence == 'exit()':
            go = False

        dialogs: List[Dialog] = [
        [{"role": "user", "content": sentence}],
        ]
        results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        )

        print(f"{results['generation']['role'].capitalize()}:{result['generation']['content']}")

    print("The end of program...")




    










if __name__ == "__main__":
    fire.Fire(main)




# 





