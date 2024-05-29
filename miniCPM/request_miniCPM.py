import requests
import json
max_length = 512
def get_completion(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {"prompt": prompt,"max_length":max_length}
    response = requests.post(url='http://172.17.1.189:6006', 
                             headers=headers, 
                             data=json.dumps(data))
    # print(response)
    return response.json()['response']

if __name__ == '__main__':
    print(get_completion("山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？"))