from transformers import AutoTokenizer
from ctypes_bindings import *
import threading
import time
import sys
import os

MODEL_PATH = "./models"

# Create a dict of various model configs, and then check the ./models directory if any exist
# This will become the content of the model selector's drop down menu
def available_models():
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    # Initialize the dict of available models as empty
    rkllm_model_files = {}
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
    # Populate the dictionary with found models, and their base configurations
    for family, config in model_configs.items():
            for model, details in config["models"].items():
                filename = details["filename"]
                if os.path.exists(os.path.join(MODEL_PATH, filename)):
                    rkllm_model_files[model] = {}
                    rkllm_model_files[model].update({"name": model,"family": family, "filename": filename, "config": config["base_config"]})
    return rkllm_model_files

# Define the callback function
def callback_impl(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        print("\n")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        print("run error")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_GET_LAST_HIDDEN_LAYER:
        '''
        If using the GET_LAST_HIDDEN_LAYER function, the callback interface will return the memory pointer: last_hidden_layer, the number of tokens: num_tokens, and the size of the hidden layer: embd_size.
        With these three parameters, you can retrieve the data from last_hidden_layer.
        Note: The data needs to be retrieved during the current callback; if not obtained in time, the pointer will be released by the next callback.
        '''
        if result.last_hidden_layer.embd_size != 0 and result.last_hidden_layer.num_tokens != 0:
            data_size = result.last_hidden_layer.embd_size * result.last_hidden_layer.num_tokens * ctypes.sizeof(ctypes.c_float)
            print(f"data_size: {data_size}")
            global_text.append(f"data_size: {data_size}\n")
            output_path = os.getcwd() + "/last_hidden_layer.bin"
            with open(output_path, "wb") as outFile:
                data = ctypes.cast(result.last_hidden_layer.hidden_states, ctypes.POINTER(ctypes.c_float))
                float_array_type = ctypes.c_float * (data_size // ctypes.sizeof(ctypes.c_float))
                float_array = float_array_type.from_address(ctypes.addressof(data.contents))
                outFile.write(bytearray(float_array))
                print(f"Data saved to {output_path} successfully!")
                global_text.append(f"Data saved to {output_path} successfully!")
        else:
            print("Invalid hidden layer data.")
            global_text.append("Invalid hidden layer data.")
        global_state = state
        time.sleep(0.05)
        sys.stdout.flush()
    else:
        # Save the output token text and the RKLLM running state
        global_state = state
        # Monitor if the current byte data is complete; if incomplete, record it for later parsing
        try:
            global_text.append((split_byte_data + result.contents.text).decode('utf-8'))
            print((split_byte_data + result.contents.text).decode('utf-8'), end='')
            split_byte_data = bytes(b"")
        except:
            split_byte_data += result.contents.text
        sys.stdout.flush()

# Connect the callback function between the Python side and the C++ side
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

class RKLLMLoaderClass:
    def __init__(self, model="", qtype="w8a8", opt="1", hybrid_quant="1.0"):        
        self.qtype = qtype
        self.opt = opt
        self.model = model
        self.hybrid_quant = hybrid_quant
        if self.model == "":
            print("No models loaded yet!")
        else:
            self.available_models = available_models()
            self.model = self.available_models[model]
            self.family = self.model["family"]
            self.model_path = "models/" + self.model["filename"]
            self.base_config = self.model["config"]
            self.model_name = self.model["name"]
            self.st_model_id = self.base_config["st_model_id"]
            self.system_prompt = self.base_config["system_prompt"]
            self.rkllm_param = RKLLMParam()
            self.rkllm_param.model_path = bytes(self.model_path, 'utf-8')
            self.rkllm_param.max_context_len = self.base_config["max_context_len"]
            self.rkllm_param.max_new_tokens = self.base_config["max_new_tokens"]
            self.rkllm_param.skip_special_token = True
            self.rkllm_param.top_k = self.base_config["top_k"]
            self.rkllm_param.top_p = self.base_config["top_p"]
            self.rkllm_param.temperature = self.base_config["temperature"]
            self.rkllm_param.repeat_penalty = self.base_config["repeat_penalty"]
            self.rkllm_param.frequency_penalty = self.base_config["frequency_penalty"]
            self.rkllm_param.presence_penalty = 0.0
            self.rkllm_param.mirostat = 0
            self.rkllm_param.mirostat_tau = 5.0
            self.rkllm_param.mirostat_eta = 0.1
            self.rkllm_param.is_async = False
            self.rkllm_param.img_start = "".encode('utf-8')
            self.rkllm_param.img_end = "".encode('utf-8')
            self.rkllm_param.img_content = "".encode('utf-8')
            self.rkllm_param.extend_param.base_domain_id = 0
            self.handle = RKLLM_Handle_t()
            self.rkllm_init = rkllm_lib.rkllm_init
            self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
            self.rkllm_init.restype = ctypes.c_int
            self.rkllm_init(self.handle, self.rkllm_param, callback)
            self.rkllm_run = rkllm_lib.rkllm_run
            self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
            self.rkllm_run.restype = ctypes.c_int
            self.rkllm_abort = rkllm_lib.rkllm_abort
            self.rkllm_abort.argtypes = [RKLLM_Handle_t]
            self.rkllm_abort.restype = ctypes.c_int
            self.rkllm_destroy = rkllm_lib.rkllm_destroy
            self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
            self.rkllm_destroy.restype = ctypes.c_int

    # Record the user's input prompt
    def get_user_input(self, user_message, history):
        history = history + [[user_message, None]]
        return "", history
    def tokens_to_ctypes_array(self, tokens, ctype):
        # Converts a Python list to a ctypes array.
        # The tokenizer outputs as a Python list.
        return (ctype * len(tokens))(*tokens)
    # Run inference
    def run(self, prompt):       
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_input = RKLLMInput()
        self.rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_TOKEN
        self.rkllm_input.input_data.token_input.input_ids = self.tokens_to_ctypes_array(prompt, ctypes.c_int)
        self.rkllm_input.input_data.token_input.n_tokens = ctypes.c_ulong(len(prompt))
        self.rkllm_run(self.handle, ctypes.byref(self.rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
        return
    # Release RKLLM object from memory
    def release(self):
        self.rkllm_abort(self.handle)
        self.rkllm_destroy(self.handle)
    # Retrieve the output from the RKLLM model and print it in a streaming manner
    def get_RKLLM_output(self, history):
        # Link global variables to retrieve the output information from the callback function
        global global_text, global_state
        global_text = []
        global_state = -1
        # Build prompt and tokenize. 
        user_message = history[-1][0]
        # Gemma 2 does not support system prompt.
        if self.system_prompt == "":
            chat = [
                {"role": "user", "content": user_message}
            ]
        else:
            chat = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
        tokenizer = AutoTokenizer.from_pretrained(self.st_model_id, trust_remote_code=True)
        prompt = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
        # Create a thread for model inference
        model_thread = threading.Thread(target=self.run, args=(prompt,))
        model_thread.start()
        # history[-1][1] represents the current dialogue
        history[-1][1] = ""
        # Wait for the model to finish running and periodically check the inference thread of the model
        model_thread_finished = False
        while not model_thread_finished:
            while len(global_text) > 0:
                history[-1][1] += global_text.pop(0)
                time.sleep(0.005)
                # Gradio automatically pushes the result returned by the yield statement when calling the then method
                yield history
            model_thread.join(timeout=0.005)
            model_thread_finished = not model_thread.is_alive()