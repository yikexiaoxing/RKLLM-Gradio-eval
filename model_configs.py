model_configs = {
    "Llama-3.2-Instruct": {
        "base_config": {
            "st_model_id": "c01zaut/Llama-3.2-3B-Instruct-rk3588-1.1.1",
            "max_context_len": 4096,
            "max_new_tokens": 8192,
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.6,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.3,
            "system_prompt": "You are Llama 3.2, an artificial intelligence model trained by Meta. You are a helpful assistant."
            },
        "models": {
            "Llama-3.2-Instruct-3B": {"filename": "Llama-3.2-3B-Instruct-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"},
            "Llama-3.2-Instruct-1B": {"filename": "Llama-3.2-1B-Instruct-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"}
        }
    },
    "LLaMA-Mesh": {
        "base_config": {
            "st_model_id": "c01zaut/LLaMA-Mesh-rk3588-1.1.1",
            "max_context_len": 4096,
            "max_new_tokens": 131072,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 0.6,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.3,
            "system_prompt": "" 
        },
        "models": {
            "LLaMA-Mesh-w8a8-opt-hybrid": {"filename": "LLaMA-Mesh-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"},
            "LLaMA-Mesh-w8a8_g256-opt": {"filename": "models/LLaMA-Mesh-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.0.rkllm"},
            "LLaMA-Mesh-w8a8-opt": {"filename": "LLaMA-Mesh-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm"},
            "LLaMA-Mesh-w8a8_g512-hybrid": {"filename": "LLaMA-Mesh-rk3588-w8a8_g512-opt-0-hybrid-ratio-1.0.rkllm"},
            "LLaMA-Mesh-w8a8_g128-opt-hybrid-0.5": {"filename": "LLaMA-Mesh-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.5.rkllm"}
        }
    },
    "Phi-3.5-Mini-Instruct": {
        "base_config": {
            "st_model_id": "microsoft/Phi-3.5-mini-instruct",
            "max_context_len": 4096,
            "max_new_tokens": 512,
            "top_k": 8,
            "top_p": 0.8,
            "temperature": 0.7,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.5,
            "system_prompt": "You are Phi 3.5 Mini, an artificial intelligence model trained by Microsoft. You are a helpful AI assistant."
        },
        "models": {
            "Phi-3.5-Mini-Instruct": {"filename": "Phi-3.5-mini-instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm"},
            "Phi-3.5-Mini-Instruct-w8a8_g256-opt": {"filename": "Phi-3.5-mini-instruct-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.5.rkllm"},
            "Phi-3.5-Mini-Instruct-w8a8_g512-opt": {"filename": "Phi-3.5-mini-instruct-rk3588-w8a8_g512-opt-1-hybrid-ratio-0.5.rkllm"}
        }
    },
    "Phi-3-Mini-Instruct": {
        "base_config": {
            "st_model_id": "c01zaut/Phi-3-mini-128k-instruct-rk3588-1.1.2",
            "max_context_len": 4096,
            "max_new_tokens": 4096,
            "top_k": 8,
            "top_p": 0.8,
            "temperature": 0.7,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.5,
            "system_prompt": "You are Phi 3.5 Mini, an artificial intelligence model trained by Microsoft. You are a helpful AI assistant."
        },
        "models": {
            "Phi-3-mini-128k-instruct-w8a8": {"filename": "Phi-3-mini-128k-instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm"},
            "Phi-3-mini-128k-instruct-w8a8-opt": {"filename": "Phi-3-mini-128k-instruct-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm"},
            "Phi-3-mini-128k-instruct-w8a8-opt-hybrid-0.5": {"filename": "Phi-3-mini-128k-instruct-rk3588-w8a8-opt-1-hybrid-ratio-0.5.rkllm"}
        }
    },
    "Qwen-2.5-Instruct": {
        "base_config": {
            "st_model_id": "Qwen/Qwen2.5-14B-Instruct",
            "max_context_len": 4096,
            "max_new_tokens": 8192,
            "top_k": 5,
            "top_p": 0.8,
            "temperature": 0.2,
            "repeat_penalty": 1.00,
            "frequency_penalty": 0.2,
            "system_prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        },
        "models": {
            "Qwen2.5-1.5B-Instruct": {"filename": "Qwen2.5-1.5B-Instruct-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"},
            "Qwen2.5-3B-Instruct": {"filename": "Qwen2.5-3B-Instruct-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"},
            "Qwen2.5-7B-Instruct": {"filename": "Qwen2.5-7B-Instruct-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"},
            "Qwen2.5-Coder-7B-Instruct": {"filename": "Qwen2.5-Coder-7B-Instruct-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.5.rkllm"},
            "Qwen2.5-Coder-1.5B-Instruct-w8a8-hybrid": {"filename": "Qwen2.5-Coder-1.5B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-1.0.rkllm"}
        },
    },
    "Gemma-2-LongCoT": {
        "base_config": {
            "st_model_id": "c01zaut/OpenLongCoT-Base-Gemma2-2B-rk3588-1.1.1",
            "max_context_len": 4096,
            "max_new_tokens": 4096,
            "top_k": 1,
            "top_p": 0.8,
            "temperature": 0.1,
            "repeat_penalty": 1.05,
            "frequency_penalty": 0.5,
            "system_prompt": ""
        },
        "models": {
            "OpenLongCoT-Base-Gemma2-2B-rk3588-1.1.1": {"filename": "OpenLongCoT-Base-Gemma2-2B-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"}
        }
    },
    "Gemma-2-IT-Google": {
        "base_config": {
            "st_model_id": "c01zaut/gemma-2-9b-it-rk3588-1.1.1",
            "max_context_len": 4096,
            "max_new_tokens": 8192,
            "top_k": 15,
            "top_p": 0.95,
            "temperature": 0.7,
            "repeat_penalty": 1.05,
            "frequency_penalty": 0.5,
            "system_prompt": ""
        },
        "models": {
            "gemma-2-2b-it-opt": {"filename": "gemma-2-2b-it-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"},
            "gemma-2-9b-it-opt": {"filename": "gemma-2-9b-it-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"},
            "codegemma-7b-it-opt": {"filename": "codegemma-7b-it-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm"},
            "codegemma-7b-it-opt-hybrid": {"filename": "codegemma-7b-it-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"}
        }
    },
    "InternLM2_5": {
        "base_config": {
            "st_model_id": "internlm/internlm2_5-7b-chat",
            "max_context_len": 4096,
            "max_new_tokens": 8192,
            "top_k": 30,
            "top_p": 0.8,
            "temperature": 0.5,
            "repeat_penalty": 1.0005,
            "frequency_penalty": 0.2,
            "system_prompt": "You are InternLM (书生·浦语), a helpful, honest, and harmless AI assistant developed by Shanghai AI Laboratory (上海人工智能实验室)."    
        },
        "models": {
            "internlm2_5-1_8b-chat-w8a8_g512-opt": {"filename": "internlm2_5-1_8b-chat-rk3588-w8a8_g512-opt-1-hybrid-ratio-1.0.rkllm"},
            "internlm2_5-1_8b-chat-w8a8": {"filename": "internlm2_5-1_8b-chat-rk3588-w8a8-opt-0-hybrid-ratio-1.0.rkllm"},
            "internlm2_5-1_8b-chat-w8a8-opt": {"filename": "internlm2_5-1_8b-chat-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"},
            "internlm2_5-20b-chat-w8a8-opt": {"filename": "internlm2_5-20b-chat-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"},
            "internlm2_5-20b-chat-w8a8_g512-opt": {"filename": "internlm2_5-20b-chat-rk3588-w8a8_g512-opt-1-hybrid-ratio-1.0.rkllm"},
            "internlm2_5-20b-chat-w8a8_g512": {"filename": "internlm2_5-20b-chat-rk3588-w8a8_g512-opt-0-hybrid-ratio-1.0.rkllm"},
            "internlm2_5-7b-chat-w8a8-opt": {"filename": "internlm2_5-7b-chat-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"}
        }
    }
}