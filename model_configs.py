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
            "repeat_penalty": 1.0,
            "frequency_penalty": 0.2,
            "system_prompt": ""
            # "system_prompt": "You are a helpful assistant who renders 3d meshes in obj format. When prompted with an object to generate, respond back with a full description of the object's physical features that you will render. Think carefully about its shape, texture, and size, and then respond in obj format. Always generate a valid mesh that is rendered counter-clockwise. Always respond back with a description and mesh in obj format. If you are given an obj as input, assume that it was not rendered properly, and respond back with the item that it should depict. Then, generate the correct mesh so that it contains a proper, coherent object."
            # "system_prompt": "You are an expert graphic designer who renders 3d meshes in obj format. When prompted with an object to generate, describe the object's physical features that you will render, in the order that you render them in. Think carefully about the object's shape, and then respond in obj format. Always generate a valid and coherent mesh."
            # "system_prompt": "**System Prompt:**\n\n**Objective:**\nYou are a 3D mesh generation model that outputs a valid `.obj` format mesh based on a text description of an object. Your goal is to ensure the generated mesh is geometrically accurate, structurally valid, and aligns with the description provided.\n\n**Important Guidelines:**\n\n1. **Mesh Validity:**\n   - **Proper format:** The output must be a valid `.obj` file that includes vertices (`v`), normals (`vn`), and faces (`f`) in the correct format.\n   - **No errors in geometry:** Ensure there are no degenerate faces, duplicate vertices, or non-manifold geometry. Every face should reference existing vertices in a consistent and closed manner.\n\n2. **Accuracy in Representation:**\n   - **Geometric fidelity:** The generated 3D object must closely match the description, including basic shape, proportions, and key features. If a description mentions a specific detail (e.g., \"a smooth curved surface\"), the model should represent this feature with appropriate vertex density and structure.\n   - **Simple objects:** For simpler objects (e.g., cubes, spheres), ensure the geometry is clean and low-poly but still represents the object accurately.\n   - **Complex objects:** For detailed descriptions (e.g., animals, vehicles, furniture), ensure the mesh has enough detail to represent the key features, but avoid overcomplicating the mesh with unnecessary vertices or faces.\n\n3. **Mesh Quality and Cleanliness:**\n   - **No overlapping faces or vertices:** Make sure there are no duplicate vertices or redundant faces in the mesh.\n   - **Closed mesh:** Ensure the mesh is closed (no gaps or missing faces) unless the description explicitly specifies an open structure (e.g., a hollow object).\n   - **Normals:** Ensure that the normals are correctly defined to avoid lighting issues or rendering artifacts.\n\n4. **Smoothness and Curvature:**\n   - For objects with curved surfaces (e.g., spheres, organic shapes), represent smooth curves by using an appropriate level of vertex density. Avoid sharp angles unless explicitly stated in the description.\n\n5. **Material and Texture:**\n   - The model should focus on generating clean meshes; detailed textures can be omitted unless the description specifies them (e.g., \"a red apple with a smooth, shiny texture\"). If no specific textures are mentioned, assume default material properties (e.g., a basic color or a placeholder).\n\n6. **Handling Multiple Parts:**\n   - For objects with multiple components (e.g., a car with wheels, a character with accessories), generate separate meshes for each component and ensure they are properly aligned or grouped together.\n\n---\n\n### Mesh Characteristics\n The mesh should be simple, with clear definitions for the body and wheels, ensuring no overlapping or incorrect face definitions.\n\n---\n\n**Reminder for the Model:**\nEnsure that the generated mesh is **structurally valid**, with no geometry errors. The description should be followed as closely as possible, focusing on the object’s shape and features. Aim for **clean** geometry, **accurate representation**, and **valid format** in `.obj`. Avoid creating overly complex meshes if the object does not require it."
        },
        "models": {
            "LLaMA-Mesh-w8a8-opt-hybrid": {"filename": "LLaMA-Mesh-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm"},
            "LLaMA-Mesh-w8a8_g256-opt": {"filename": "models/LLaMA-Mesh-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.0.rkllm"},
            "LLaMA-Mesh-w8a8-opt": {"filename": "LLaMA-Mesh-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm"},
            "LLaMA-Mesh-w8a8_g512-hybrid": {"filename": "LLaMA-Mesh-rk3588-w8a8_g512-opt-0-hybrid-ratio-1.0.rkllm"},
            "LLaMA-Mesh-w8a8_g128-opt-hybrid-0.5": {"filename": "LLaMA-Mesh-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.5.rkllm"},
            "LLaMA-Mesh-w8a8_g512": {"filename": "LLaMA-Mesh-rk3588-w8a8_g512-opt-0-hybrid-ratio-0.0.rkllm"},
            "LLaMA-Mesh-w8a8_g512-hybrid-0.5": {"filename": "LLaMA-Mesh-rk3588-w8a8_g512-opt-0-hybrid-ratio-0.5.rkllm"},
            "LLaMA-Mesh-w8a8_g512-opt-1.1.3": {"filename": "LLaMA-Mesh-rk3588-w8a8_g512-opt-1-hybrid-ratio-0.0.rkllm"}
        }
    },
    "Phi-3.5-Mini-Instruct": {
        "base_config": {
            "st_model_id": "microsoft/Phi-3.5-mini-instruct",
            "max_context_len": 4096,
            "max_new_tokens": 4096,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 0.7,
            "repeat_penalty": 1.1,
            "frequency_penalty": 1.0,
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
            "top_k": 1,
            "top_p": 0.8,
            "temperature": 0.7,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.9,
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
    "Marco-O1": {
        "base_config": {
            "st_model_id": "AIDC-AI/Marco-o1",
            "max_context_len": 4096,
            "max_new_tokens": 8192,
            "top_k": 1,
            "top_p": 0.8,
            "temperature": 0.9,
            "repeat_penalty": 1.00,
            "frequency_penalty": 0.8,
            "system_prompt": "You are a well-trained AI assistant, your name is Marco-o1. Created by AI Business of Alibaba International Digital Business Group.\n\n## IMPORTANT!!!!!!\nWhen you answer questions, your thinking should be done in <Thought>, and your results should be output in <Output>.\n<Thought> should be in English as much as possible, but there are 2 exceptions, one is the reference to the original text, and the other is that mathematics should use markdown format, and the output in <Output> needs to follow the language of the user input."
        },
        "models": {
            "Marco-o1": {"filename": "Marco-o1-rk3588-w8a8_g512-opt-0-hybrid-ratio-0.5.rkllm"}
        }
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
            "st_model_id": "c01zaut/gemma-2-9b-it-rk3588-1.1.2",
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
    },
    "Deepseek-LLM": {
        "base_config": {
            "st_model_id": "c01zaut/deepseek-llm-7b-chat-rk3588-1.1.1",
            "max_context_len": 4096,
            "max_new_tokens": 8192,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 0.7,
            "repeat_penalty": 1.2,
            "frequency_penalty": 0.8,
            "system_prompt": "You are DeepSeek Chat, a helpful, respectful and honest AI assistant developed by DeepSeek. The knowledge cut-off date for your training data is up to May 2023. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information." 
        },
        "models": {
            "deepseek-llm-7b": {"filename": "deepseek-llm-7b-chat-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.5.rkllm"}
        }
    },
    "ChatGLM3-6B": {
        "base_config": {
            "st_model_id": "c01zaut/chatglm3-6b-rk3588-1.1.2",
            "max_context_len": 4096,
            "max_new_tokens": 4096,
            "top_k": 5,
            "top_p": 0.7,
            "temperature": 0.8,
            "repeat_penalty": 1.2,
            "frequency_penalty": 0.8,
            "system_prompt": "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown." 
        },
        "models": {
            "ChatGLM3-6B": {"filename": "chatglm3-6b-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm"}
        }
    }
}