from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

# 将json文件转换为csv文件
df = pd.read_json("./huanhuan.json")
ds = Dataset.from_pandas(df)
# print(ds[:3])
"""
{'instruction': ['小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——', '这个温太医啊，也是古怪，谁不知太医不得皇命不能为皇族以外的人请脉诊病，他倒好，十天半月便往咱们府里跑。', '嬛妹妹，刚刚我去府上请脉，听甄伯母说你来这里进香了。'], 
'input': ['', '', ''], 
'output': ['嘘——都说许愿说破是不灵的。', '你们俩话太多了，我该和温太医要一剂药，好好治治你们。', '出来走走，也是散心。']}
"""

# 处理数据集
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen1.5-7B-Chat/", 
                                          use_fast=False, 
                                          trust_remote_code=True)
# print(tokenizer)


def precess_func(example):
    # LLama分词器会将一个中文字切分为多个token， 因此需要开放一些最大长度， 保证数据的完整性
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    # add_special_tokens 不在开头加 special_tokens
    instruction = tokenizer(f"<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False) 

    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 因为eos token咱们也是要关注的， 所以补充为1
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }


tokenized_id = ds.map(precess_func, remove_columns=ds.column_names)
# print(tokenized_id)
"""
Dataset({
    features: ['input_ids', 'attention_mask', 'labels'],
    num_rows: 3729
})
"""

# print(tokenizer.decode(tokenized_id[0]['input_ids']))
"""
<|im_start|>system
现在你要扮演皇帝身边的女人--甄嬛<|im_end|>
<|im_start|>user
小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——<|im_end|>
<|im_start|>assistant
嘘——都说许愿说破是不灵的。<|endoftext|>
"""

# print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"]))))
"""
你们俩话太多了，我该和温太医要一剂药，好好治治你们。<|endoftext|>
"""

# 创建模型
import torch
model = AutoModelForCausalLM.from_pretrained('./qwen/Qwen1.5-7B-Chat/', 
                                             device_map="auto", 
                                             torch_dtype=torch.bfloat16)
# print(model)
"""
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 4096)
    (layers): ModuleList(
      (0-31): 32 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=True)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=True)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm()
        (post_attention_layernorm): Qwen2RMSNorm()
      )
    )
    (norm): Qwen2RMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
)
"""
# 开启梯度检查点时， 要执行该方法
model.enable_input_require_grads()
# print(model.dtype)
"""
torch.bfloat16
"""

# Lora
from peft import  LoraConfig, TaskType, get_peft_model
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], 
    inference_mode=False,  # 训练模式
    r = 8,  # Lora 秩
    lora_alpha = 32,   # Lora alpah， 具体用法参见lora原理
    lora_dropout=0.1  # dropout失活比例
)
# print(config)
"""
LoraConfig(peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, 
base_model_name_or_path=None, revision=None, 
task_type=<TaskType.CAUSAL_LM: 'CAUSAL_LM'>, 
inference_mode=False, r=8, 
target_modules={'up_proj', 'q_proj', 'k_proj', 'down_proj', 'o_proj', 'gate_proj', 'v_proj'}, 
lora_alpha=32, lora_dropout=0.1, fan_in_fan_out=False, bias='none', 
use_rslora=False, modules_to_save=None, init_lora_weights=True, 
layers_to_transform=None, layers_pattern=None, rank_pattern={}, 
alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', 
loftq_config={}, use_dora=False)
"""

model = get_peft_model(model, config)
# print(config)
"""
LoraConfig(peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, 
base_model_name_or_path='./qwen/Qwen1.5-7B-Chat/', revision=None, 
task_type=<TaskType.CAUSAL_LM: 'CAUSAL_LM'>, inference_mode=False, r=8, 
target_modules={'up_proj', 'q_proj', 'o_proj', 'k_proj', 'gate_proj', 'down_proj', 'v_proj'},
lora_alpha=32, lora_dropout=0.1, fan_in_fan_out=False, bias='none', use_rslora=False, 
modules_to_save=None, init_lora_weights=True, layers_to_transform=None, 
layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, 
megatron_core='megatron.core', 
loftq_config={}, use_dora=False)
"""
# print(model.print_trainable_parameters())
"""
trainable params: 19,988,480 || all params: 7,741,313,024 || trainable%: 0.2582052933143348
"""

# 配置训练参数
args = TrainingArguments(
    output_dir="./output/Qwen1.5", 
    per_device_train_batch_size = 4, 
    gradient_accumulation_steps = 4, 
    logging_steps=10, 
    num_train_epochs=3, 
    save_steps=100, 
    learning_rate=1e-4, 
    save_on_each_node=True, 
    gradient_checkpointing=True
)

trainer = Trainer(
    model = model, 
    args = args, 
    train_dataset=tokenized_id, 
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()








































