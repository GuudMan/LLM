import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download("01ai/Yi-6B-Chat", 
                              cache_dir="/root/autodl-tmp", 
                              revision='master')









