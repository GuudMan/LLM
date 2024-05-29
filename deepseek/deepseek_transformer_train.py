# 导入torch库，用于深度学习相关操作
import torch
# 从transformers库导入所需的类  
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig 

# 将模型路径设置为刚刚下载的模型路径
model_name = "/root/autodl-tmp/deepseek-ai/deepseek-coder-6.7b-instruct"

# 加载分词器，trust_remote_code=True允许加载远程代码
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          trust_remote_code=True)

# 加载语言模型，设置数据类型为bfloat16以优化性能（以免爆显存），并自动选择GPU进行推理
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             torch_dtype=torch.bfloat16, 
                                             device_map="auto", 
                                             trust_remote_code=True)

# 加载并设置生成配置，使用与模型相同的设置
model.generation_config = GenerationConfig.from_pretrained(model_name, 
                                                           trust_remote_code=True)

# 将填充令牌ID设置为与结束令牌ID相同，用于生成文本的结束标记
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# 定义输入消息，模型使用apply_chat_template进行消息输入，模拟用户与模型的交互
messages = [
    {"role": "user", "content": "你是谁"}
]

# 处理输入消息，并添加生成提示
input_tensor = tokenizer.apply_chat_template(messages, 
                                             add_generation_prompt=True, 
                                             return_tensors="pt")

# 使用模型生成回应，设置max_new_tokens数量为100（防止爆显存）
# 也可以将max_new_tokens设置的更大，但可能爆显存
outputs = model.generate(input_tensor.to(model.device), 
                         max_new_tokens=100)

# 模型输出，跳过特殊令牌以获取纯文本结果
result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], 
                          skip_special_tokens=True)

# 显示生成的回答
print(result)