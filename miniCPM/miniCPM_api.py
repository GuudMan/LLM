from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import uvicorn
import json
import datetime
import torch

# 设置设备参数
DEVICE = "cuda"  # 使用cuda
DEVICE_ID = "0"  # 
CUDA_DEVICE = F"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

# 创建FastAPI应用
app = FastAPI()
@app.post("/")
async def create_item(request:Request):
    global model, tokenizer  # 声明全局变量以便在函数内部使用模型或分词器
    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    prompt = json_post_list.get('prompt')  # 获取请求中的输入prompt

    # 构建输入， 并设置生成参数， temperature, 
    # top_P值和repetition_penalty(重复惩罚因子)等, 可执行修改
    responds, history = model.chat(tokenizer, 
                                   prompt, 
                                   temperature=0.5, 
                                   top_p = 0.8, 
                                   repetition_penalty=1.02)
    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串

    # 构建响应json
    answer = {
        "response": responds, 
        "status": 200, 
        "time": time
    }

    # 构建日志信息
    log = "[" + time + "]" + '", prompt:"' + prompt + '", response:"' + repr(responds) + '"'
    print(log)  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return answer  # 返回响应

# 主函数入口
if __name__ == '__main__':
    torch.manual_seed(0)  # 设置随机种子确保结果的可复现性
    # 设置模型路径
    # path = "/root/autodl-tmp/OpenBMB/MiniCPM-2B-sft-fp32"
    path = "./output/MiniCPM"
    # 从模型路径加载分词器
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 从模型路径加载模型，设置为使用bfloat16精度以优化性能，
    # 并将模型部署到支持CUDA的GPU上,trust_remote_code=True允许加载远程代码
    model = AutoModelForCausalLM.from_pretrained(path, 
                                                 torch_dtype=torch.bfloat16, 
                                                 device_map='cuda', 
                                                 trust_remote_code=True)
    model.eval()  # 设置模型为评估模式
    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(app, host='172.17.1.189', port=6006, workers=1)  # 在指定端口和主机上启动应用
    # uvicorn.run(app, host='0.0.0.0', port=6006, workers=1) 
