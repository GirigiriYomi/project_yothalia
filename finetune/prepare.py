import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig


nf8_config = BitsAndBytesConfig(
   load_in_8bit=True,
   bnb_8bit_quant_type="nf8",
   bnb_8bit_use_double_quant=True,
   bnb_8bit_compute_dtype=torch.bfloat16
)

config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.02,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
    )




tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b", 
                                          quantization_config=nf8_config,
                                          trust_remote_code=True,
                                          cache_dir='../yothalia/server/model_weights/internlm/internlm-chat-7b')

model = AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-7b", 
                                             torch_dtype=torch.float16, 
                                             trust_remote_code=True,
                                             cache_dir='../yothalia/server/model_weights/internlm/internlm-chat-7b')


special_tokens_dict = {'additional_special_tokens': 
                       ['<<SYS>>','<</SYS>>','[INST]','[/INST]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


tokenizer.save_pretrained('../yothalia/server/model_weights/internlm/internlm-chat-7b-finetune')

model.save_pretrained('../yothalia/server/model_weights/internlm/internlm-chat-7b-finetune')


model = get_peft_model(model, config)
model.print_trainable_parameters()

model.save_pretrained("../yothalia/server/model_weights/internlm/internlm-chat-7b-finetune-lora")

