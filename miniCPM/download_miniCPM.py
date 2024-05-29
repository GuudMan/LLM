import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('OpenBMB/miniCPM-bf32', 
                              cache_dir='/root/autodl-tmp', 
                              revision='master')