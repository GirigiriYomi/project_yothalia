import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer

access_token = 'YOUR_OWN_HUGGIN_FACE_TOKEN'
model_dir = "meta-llama/Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_dir, token=access_token)
tokenizer = LlamaTokenizer.from_pretrained(model_dir, token=access_token)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    return_full_text=False,
)


def llama_request(payload):
    sequences = pipeline(
        payload,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,
    )
    return sequences[0]['generated_text']