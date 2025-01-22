import csv
import requests
import argparse
from tqdm import tqdm
# 读取CSV文件的两列
def read_csv(csv_file):
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = [(row[0], row[1]) for row in reader]  # 读取问题列和标准答案列
    return data

# 发送请求到模型并获取回答
def get_model_response(prompt, model_name):
    messages = [{"role": "user", "content": prompt}]
    prompt_headers = {"Authorization": "Bearer your_api_key_here"}
    stream_mode = False
    data = {
        "model": model_name,  # 使用参数化的模型名称
        "stream": stream_mode,
        "messages": messages,
    }

    response = requests.post(
            "http://0.0.0.0:8000/v1/chat/completions",
        headers=prompt_headers,
        json=data,
        stream=stream_mode,
    )
    return response.json()

# 将结果写入CSV文件
import csv
import os
def write_results_to_csv(output_file, results, model_name):
    """
    将测试结果写入 CSV 文件。

    参数:
        output_file (str): 输出 CSV 文件的路径。
        results (list): 包含测试结果的列表，每个结果是一个字典。
        model_name (str): 当前模型的名称。
    """
    # 模型相关的列
    model_columns = [
        f"{model_name}_message",
        f"{model_name}_completion_tokens",
        f"{model_name}_generate_speed"
    ]
    #文件是否存在

    # 打开文件，追加模式
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # 如果文件不存在，写入表头
        headers = ["问题", "答案"] + model_columns
        writer.writerow(headers)

        # 写入当前模型的结果
        for result in results:
            row = [
                result.get('问题', ''),  # 获取问题，默认为空
                result.get('答案', '')   # 获取答案，默认为空
            ]
            # 添加模型相关列
            for column in model_columns:
                row.append(result.get(column, ''))  # 获取模型相关列，默认为空
            writer.writerow(row)




# 主函数
def main(input_csv, output_csv, model_names):
    data = read_csv(input_csv)  # 读取输入CSV文件
    #如果不存在输出文件，则创建

        

    results = []
    for question, answer in data[1:]:
        result = {
            '问题': question,
            '答案': answer
        }
        results.append(result)

    # 依次测试每个模型
    for model_name in model_names:
        for result in tqdm(results):
            prompt = result['问题']
            response = get_model_response(prompt, model_name)
            message = response['choices'][0]['message']['content']

            completion_tokens = response['usage']['completion_tokens']
            generate_speed = response['usage']['generate_speed']

            # 将模型结果添加到当前行
            # result[model_name] = model_name
            result[f"{model_name}_message"] = message
            result[f"{model_name}_completion_tokens"] = completion_tokens
            result[f"{model_name}_generate_speed"] = generate_speed
            print(result)

        # 将当前模型的结果写入CSV文件
        write_results_to_csv(output_csv, results, model_name)

    print(f"Results have been written to {output_csv}")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Process CSV and get model responses.")
    parser.add_argument('--input_csv', type=str, default="./eval/input_100.csv", help="Path to the input CSV file.")
    parser.add_argument('--output_csv', type=str, default="./eval/output/Llama-3.2-Instruct-3B-2.csv", help="Path to the output CSV file.")
    parser.add_argument('--model_names', type=list, default=["Llama-3.2-Instruct-3B"],  help="Names of the models to test.")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args.input_csv, args.output_csv, args.model_names)
