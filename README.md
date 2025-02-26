# Rockchip RK3588启动OpenAI接口使用说明 📄

---

## 目录 📑

1. [简介](#1-简介)
2. [环境准备](#2-环境准备)
   - [硬件要求](#21-硬件要求)
   - [软件要求](#22-软件要求)
3. [环境安装](#3-环境安装)
   - [安装Python环境](#31-安装python环境)
4. [关键配置](#4-关键配置)
   - [模型存放与配置](#41-模型存放与配置)
5. [模型列表](#5-模型列表)
6. [启动OpenAI服务](#6-启动openai服务)
7. [测试](#7-测试)
   - [使用CURL命令调试](#71-使用curl命令调试)
   - [Python示例](#72-python示例)
8. [更多资源](#8-更多资源)

---

## 1. 简介 🚀

本文档详细介绍了如何在RK3588开发板上启动OpenAI接口，并使用默认的RKLLM库（版本1.1.4）。通过本文档，您将能够快速配置环境并启动OpenAI接口服务。

**RKLLM-OpenAI** 是一个基于OpenAI接口，支持对话（chat）、补全（completions）等功能的大型语言模型。通过RKLLM，您可以轻松集成强大的自然语言处理能力到您的应用中。

---

## 2. 环境准备 🛠️

### 2.1 硬件要求
- **RK3588开发板**：确保开发板已正确连接电源、网络及其他外设。
- **存储空间**：请根据使用模型的大小设置存储空间，建议至少预留2GB空间。

### 2.2 软件要求
- **操作系统**：RK3588开发板需运行基于Linux的系统（如烧录适配的Ubuntu或Debian）。
- **Python环境**：确保已安装Python 3.8或更高版本。
- **RKLLM库**：默认使用RKLLM库版本1.1.4。

---

## 3. 环境安装 ⚙️

### 3.1 安装Python环境
确保您的系统已安装Python 3.10或更高版本。您可以通过以下命令检查Python版本：

```bash
# 在RK3588开发板上，首先更新系统以确保所有软件包为最新版本
sudo apt update
sudo apt upgrade -y

# 安装必要的Python依赖包
sudo apt install python3-pip python3-venv -y

# 检查Python版本
python3 --version
```

---

## 4. 关键配置 🔧

### 4.1 模型存放与配置
- **models**：存放RKLLM模型。
- **model_configs.py**：该文件包含目前支持的RKLLM模型与其对应的推理参数。

**配置文件示例：**
```python
"Qwen-2.5-Instruct": {
    "base_config": {
        "st_model_id": "./tokenizers/Qwen2.5-1.5B-Instruct",
        "max_context_len": 8192,
        "max_new_tokens": 8192,
        "top_k": 20,
        "top_p": 0.9,
        "temperature": 0.6,
        "repeat_penalty": 1.1,
        "frequency_penalty": 0.3,
        "system_prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    },
    "models": {
        "Qwen2.5-1.5B-Instruct": {"filename": "Qwen2.5-1.5B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "1.5B"},
        "Qwen2.5-3B-Instruct": {"filename": "Qwen2.5-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "3B"},
        "Qwen2.5-7B-Instruct": {"filename": "Qwen2.5-7B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "7B"},
    }
}
```

---

## 5. 模型列表 📋

目前支持多种模型，您可以根据需求选择合适的模型：

- **Llama系列**：
  - Llama-3.2-Instruct-3B
  - Llama-3.2-Instruct-1B

- **Qwen系列**：
  - Qwen2.5-1.5B-Instruct
  - Qwen2.5-3B-Instruct
  - Qwen2.5-7B-Instruct

- **Phi系列**：
  - Phi-3-mini-4k-instruct
  - Phi-3-mini-128k-instruct

- **其他模型**：
  - gemma-2-2b-it-instruct
  - internlm2_5-1_8b-chat
  - deepseek-llm-7b-chat

---

## 6. 启动OpenAI服务 🚀

使用以下命令启动OpenAI服务：

```bash
python3 rkllm_server_openai.py --host 0.0.0.0 --port 8000
```

---

## 7. 测试 🧪

### 7.1 使用CURL命令调试
```bash
curl -L -X POST 'http://<host>:<port>/v1/completions' \
-H 'Content-Type: application/json' \
-H 'Accept: application/json' \
--data-raw '{
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "Hi",
      "role": "user"
    }
  ],
  "model": "Llama-3.2-Instruct-1B",
  "frequency_penalty": 0,
  "max_tokens": 2048,
  "presence_penalty": 0,
  "response_format": {
    "type": "text"
  }
}'
```

### 7.2 Python示例
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<host>:<port>/v1",
    api_key="your-api-key-here",  # ModelScope Token
)

response = client.chat.completions.create(
    model="Llama-3.2-Instruct-1B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    max_tokens=1024,
    temperature=0.7,
    stream=False
)

print(response.choices[0].message.content)
```

---

## 8. 更多资源 📚

- [RKLLM官方文档](https://github.com/airockchip/rknn-llm)
- [OpenAI官方文档](https://platform.openai.com/docs)

通过以上文档，您可以快速上手使用RKLLM的OpenAI接口，集成强大的自然语言处理能力到您的应用中。