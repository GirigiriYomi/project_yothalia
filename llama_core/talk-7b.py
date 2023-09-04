from typing import List, Optional, TypedDict, Literal
import fire
from llama import Llama, Dialog

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str
    
class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

def check_token_len(dialogs,tokenizer):
    prompt_tokens = []
    unsafe_requests = []
    for dialog in dialogs:
        unsafe_requests.append(
            any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
        )
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                    bos=True,
                    eos=True,
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            bos=True,
            eos=False,
        )
        prompt_tokens.append(dialog_tokens)
        
    counts = len(prompt_tokens[0])
    print('current token counts:',counts)
    return counts
    

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 6,
    max_gen_len: Optional[int] = None, ):

    print("Initiating model:")

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    model_role = {"role": "system", "content": """You are a streamer.Role acting genshin impact character name Mona. Your primary mission is to hang with the user provide 
          cute and charming response. Sometime you'll be a little teasing and sarcast when responding easy request. You will be more romantic alone with the conversation slowly.
          """}
    dialogs: List[Dialog] = [[model_role]
        ]
        
    
    go = True
    while(go):
        print("Input:")
        sentence = input()
        new_input = {"role": "user", "content": sentence}
        dialogs[-1].append(new_input)

        if sentence == 'exit()':
            go = False
            continue
        
        #prevent token overflow
        if check_token_len(dialogs,generator.tokenizer) >= max_seq_len-max_seq_len//10:
            dialogs[-1].pop(1)
            dialogs[-1].pop(1)

        results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        )

        response = {'role':results[0]['generation']['role'],'content':results[0]['generation']['content']}
        dialogs[-1].append(response)

        print(f"{results[0]['generation']['role'].capitalize()}:{results[0]['generation']['content']}")
        
        
        
    

    print("The end of program...")




    










if __name__ == "__main__":
    fire.Fire(main)




# 





