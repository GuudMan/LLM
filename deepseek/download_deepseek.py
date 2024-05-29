import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from modelscope import GenerationConfig
model_dir = snapshot_download('deepseek-ai/deepseek-coder-6.7b-instruct', cache_dir='/root/autodl-tmp', revision='master')































