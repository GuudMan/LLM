import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from modelscope import GenerationConfig
model_dir = snapshot_download('THUDM/chatglm2-6b', 
                              cache_dir='/root/autodl-tmp', 
                              revision='master')




















