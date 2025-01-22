model_configs = {
    "Llama-3.2-Instruct": {
        "base_config": {
            "st_model_id": "c01zaut/Llama-3.2-3B-Instruct-rk3588-1.1.1",
            "max_context_len": 4096,
            "max_new_tokens": 4096,
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.6,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.3,
            "system_prompt": "You are Llama 3.2, an artificial intelligence model trained by Meta. You are a helpful assistant."
            },
        "models": {
            "Llama-3.2-Instruct-3B": {"filename": "Llama-3.2-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "3B"}
        }
    },
    "Qwen-2.5-Instruct": {
        "base_config": {
            "st_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "max_context_len": 2048,
            "max_new_tokens": 2048,
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
            "st_model_id": "microsoft/Phi-3-mini-128k-instruct",
            "max_context_len": 2048,
            "max_new_tokens": 2048,
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.6,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.3,
            "system_prompt": "You are Phi 3.5 Mini, an artificial intelligence model trained by Microsoft. You are a helpful AI assistant."
        },
        "models": {
            "Phi-3-mini-4k-instruct": {"filename": "Phi-3-mini-4k-instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "3.8B"},
            "Phi-3-mini-8k-instruct": {"filename": "Phi-3-mini-128k-instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm", "parameter_size": "3.8B"},
        }
    },
}
