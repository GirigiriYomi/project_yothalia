#import torch
#import transformers
from llama import Llama

generator = Llama.build(
        ckpt_dir='llama-2-13b-chat',
        tokenizer_path='tokenizer.model',
        max_seq_len=256,
        max_batch_size=2,
    )
    
model = generator.model
tokenizer = generator.tokenizer

                
            
print("Done here")    
