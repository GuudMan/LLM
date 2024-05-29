from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaTokenizerFast
import torch

"""
分别重写了构造函数和_call函数： 对于构造函数， 我们在对象实例化的一开始就加载本地部署的Qwen2模型，
从而避免每一次调用都需要重新加兹安模型带来的时间过长，
_call函数是LLM类的核心函数， LangChain会调用该函数来调用LLM， 在该函数中， 我们调用已实例化模型的
chat方法， 从而实现对模型的调用并返回调用结果。
"""
class Qwen2_LLM(LLM):
    # 基于本地 Qwen2 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
        
    def __init__(self, mode_name_or_path :str):

        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)
        print("完成本地模型的加载")
        
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        messages = [{"role": "user", "content": prompt }]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(model_inputs.input_ids,max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    @property
    def _llm_type(self) -> str:
        return "Qwen2_LLM"
    

# 调用
llm = Qwen2_LLM(mode_name_or_path="/root/autodl-tmp/qwen/Qwen1.5-7B-Chat")
res = llm("时间会治愈一切")
print(res)