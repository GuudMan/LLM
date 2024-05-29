import requests

def get_audio_response(audio_file_path, text_query):
    # 设置API的URL
    url = 'http://172.17.1.189:6006/test-audio/'

    # 设置音频文件路径和文本查询的参数
    params = {
        'audio': audio_file_path,  # 音频文件路径
        'text': text_query         # 文本查询
    }

    # 发送GET请求
    response = requests.get(url, params=params)

    # 提取所需信息
    result = {
        "response": response.json(),
        "status_code": response.status_code,
        "time": response.headers.get('Date')  # 获取响应头中的时间信息
    }
    return result

if __name__ == '__main__':
    # 测试请求
    audio_file = '/root/autodl-tmp/1272-128104-0000.flac'
    text_query = '这是男生还是女生说的？'
    completion = get_audio_response(audio_file, text_query)
    print(completion)

    








