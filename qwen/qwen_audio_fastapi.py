from fastapi import FastAPI, Request, File, UploadFile, Query
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import uvicorn
import json
import datetime
import torch
import os
import nest_asyncio
nest_asyncio.apply()

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

# 创建FastAPI应用
app = FastAPI()

# 处理POST请求的端点
@app.post("/audio/")
async def create_audio_item(file: UploadFile = File(...)):
    global model, tokenizer  # 使用全局变量
    # 保存音频文件到临时路径
    temp_file_path = f"temp/{file.filename}"
    with open(temp_file_path, 'wb') as f:
        f.write(file.file.read())
    file.file.close()

    # 1st dialogue turn
    query = tokenizer.from_list_format([
        {'audio': temp_file_path},  # 使用保存的临时音频文件路径
        {'text': 'what does the person say?'},
    ])
    response, history = model.chat(tokenizer, 
                                   query=query, 
                                   history=None)

    # 清理临时文件
    os.remove(temp_file_path)

    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    print(f"[{time}] Response: {response}")  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return answer

@app.get("/test-audio/")
def test_audio(audio_file_path: str = Query('/root/autodl-tmp/1272-128104-0000.flac', alias='audio'),
               text_query: str = Query('what does the person say?', 
                                       alias='text')):
    """
    测试音频接口，用户可以指定音频文件路径和文本查询
    :param audio_file_path: 音频文件的路径
    :param text_query: 文本查询内容
    """

    # 使用model和tokenizer处理音频和文本
    query = tokenizer.from_list_format([
        {'audio': audio_file_path},
        {'text': text_query},
    ])
    response, history = model.chat(tokenizer, 
                                   query=query, 
                                   history=None)

    return {"response": response}

# 主函数入口
if __name__ == '__main__':
    mode_name_or_path = '/root/autodl-tmp/qwen/Qwen-Audio-Chat'
    # 加载预训练的分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, 
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, 
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16,  
                                                 device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()  # 设置模型为评估模式
    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(app, host='172.17.1.189', port=6006, workers=1)  # 在指定端口和主机上启动应用








