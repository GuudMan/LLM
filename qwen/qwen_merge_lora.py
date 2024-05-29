#%%
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


model_name_or_path = "/root/autodl-tmp/qwen/Qwen1.5-7B-Chat"


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                          trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, 
                                  trust_remote_code=True, 
                                  device_map='auto', 
                                  torch_dtype=torch.bfloat16)#.half().cuda()


peft_model_id = "/root/autodl-tmp/output/Qwen1.5/checkpoint-600"
model = PeftModel.from_pretrained(model, peft_model_id)
model = model.eval()

# 合并lora
model_merge = model.merge_and_unload()
merger_lora_model_path = "qwen_merge_dir"

model_merge.save_pretrained(merger_lora_model_path, max_shard_size="2GB")
tokenizer.save_pretrained(merger_lora_model_path)