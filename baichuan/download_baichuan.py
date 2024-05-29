from modelscope import snapshot_download
import torch
import os

model_dir = snapshot_download("baichuan-inc/BaiChuan2-7B-Chat", 
                              cache_dir="/root/autodl-tmp", 
                              revision="v1.0.4")