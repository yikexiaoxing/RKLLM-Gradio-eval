model_configs = {
    "Llama-3.2-Instruct": {
        "base_config": {
            "st_model_id": "./tokeniers/Llama-3.2-3B-Instruct",
            # "max_context_len": 16384, 3B
            # "max_new_tokens": 16384, 3B
            "max_context_len": 65536,
            "max_new_tokens": 65536,
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.6,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.3,
            "system_prompt": "You are Llama 3.2, an artificial intelligence model trained by Meta. You are a helpful assistant."
            },
        "models": {
            "Llama-3.2-Instruct-3B": {"filename": "Llama-3.2-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "3B"},
            "Llama-3.2-Instruct-1B": {"filename": "Llama-3.2-1B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "1B"}
        }
    },
    "Qwen-2.5-Instruct": {
        "base_config": {
            "st_model_id": "./tokeniers/Qwen2.5-1.5B-Instruct",
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
    },
    "Phi-3-mini-4k-instruct": {
        "base_config": {
            "st_model_id": "./tokeniers/Phi-3-mini-4k-instruct",
            "max_context_len": 8192,
            "max_new_tokens": 8192,
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.6,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.3,
            "system_prompt": "You are Phi 3.5 Mini, an artificial intelligence model trained by Microsoft. You are a helpful AI assistant."
        },
        "models": {
            "Phi-3-mini-4k-instruct": {"filename": "Phi-3-mini-4k-instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "3.8B"},
            "Phi-3-mini-128k-instruct": {"filename": "Phi-3-mini-128k-instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "3.8B"},
            "Phi-3-mini-instruct": {"filename": "Phi-3.5-mini-instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "3.8B"},

        }
    },
    "gemma-2-2b-it": {
        "base_config": {
            "st_model_id": "./tokeniers/gemma-2-2b-it",
            "max_context_len": 30720,
            "max_new_tokens": 30720,
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.6,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.3,
            "system_prompt": ""
        },
        "models": {
            "gemma-2-2b-it-instruct": {"filename": "gemma-2-2b-it-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "2B"},
        }
    },
    "internlm2_5-instruct": {
        "base_config": {
            "st_model_id": "internlm/internlm2_5-1_8b-chat",
            "max_context_len": 1024,
            "max_new_tokens": 1024,
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.6,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.3,
            "system_prompt": "You are internlm2_5, an artificial intelligence model trained by Microsoft. You are a helpful AI assistant."
        },
        "models": {
            "internlm2_5-1_8b-chat": {"filename": "internlm2_5-1_8b-chat-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "1.8B"},
            "internlm2_5-7b-chat-1m": {"filename": "internlm2_5-7b-chat-1m-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "7B"},
            "internlm2_5-7b": {"filename": "internlm2_5-7b-chat-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "7B"},

        }
    },
    "deepseek-llm-7b-chat": {
        "base_config": {
            "st_model_id": "./tokeniers/deepseek",
            "max_context_len": 1024,
            "max_new_tokens": 1024,
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.6,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.3,
            "system_prompt": "You are deepseek, an artificial intelligence model trained by Microsoft. You are a helpful AI assistant."
        },
        "models": {
            "deepseek-llm-7b-chat": {"filename": "deepseek-llm-7b-chat-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "1.8B"},

        }
    },
}
