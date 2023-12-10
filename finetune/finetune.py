import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from accelerate import Accelerator

from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig, prepare_model_for_kbit_training

import pandas as pd

###
### torchrun --nproc_per_node 2 finetune.py
###

nf8_config = BitsAndBytesConfig(
   load_in_8bit=True,
   bnb_8bit_quant_type="nf8",
   bnb_8bit_use_double_quant=True,
   bnb_8bit_compute_dtype=torch.bfloat16
)
accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained("../yothalia/server/model_weights/internlm/internlm-chat-7b-finetune", 
                                                load_in_4bit = True,
                                                #peft_config=config,
                                                #device_map="auto",
                                                trust_remote_code=True)
                                                
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
print(set(model.hf_device_map.values()))


tokenizer = AutoTokenizer.from_pretrained("../yothalia/server/model_weights/internlm/internlm-chat-7b-finetune",
                                            trust_remote_code=True)


peft_model_id = "../yothalia/server/model_weights/internlm/internlm-chat-7b-finetune-lora"
config = PeftConfig.from_pretrained(peft_model_id)
model = PeftModel.from_pretrained(model, peft_model_id)

model = accelerator.prepare(model)

for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True

model.print_trainable_parameters()

df = pd.read_csv('../train_sample/csv/train.csv',index_col=0)
df = df.applymap(lambda x: tokenizer(x, truncation=True)).reset_index(drop=True) # max length = 512 will let model not learn eos token
df = df.sample(frac=1).reset_index(drop=True)

df_test = df[-200:-1].reset_index(drop=True)
df_train = df[0:-200].reset_index(drop=True)


training_args = TrainingArguments(

    # Learning rate
    learning_rate=5.0e-5,

    # Number of training epochs
    num_train_epochs=3,

    # Max steps to train for (each step is a batch of data)
    # Overrides num_train_epochs, if not -1
    #max_steps=max_steps,

    # Batch size for training
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1, # Batch size for evaluation

    # Directory to save model checkpoints
    output_dir='./ckp',

    # Other arguments
    overwrite_output_dir=False, # Overwrite the content of the output directory
    disable_tqdm=False, # Disable progress bars
    eval_steps=400, # Number of update steps between two evaluations
    save_steps=400, # After # steps model is saved
    warmup_steps=1, # Number of warmup steps for learning rate scheduler
    
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    optim="adamw_torch",
    gradient_accumulation_steps = 4,
    gradient_checkpointing=False,

    # Parameters for early stopping
    load_best_model_at_end=False, # set to true will blow up cuda mem
    save_total_limit=8,
    greater_is_better=False,

)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
print('Parallel Status:',training_args.parallel_mode)


from transformers import Trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=df_train['train'],
    eval_dataset=df_test['train'],
    data_collator=data_collator,
)


trainer.train()
