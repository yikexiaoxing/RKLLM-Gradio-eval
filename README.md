# RKLLM Gradio

### This repository serves as a basic proof-of-concept for a Gradio interface for RKLLM with model switching.

## Getting started

To get started, clone this repository and enter the directory:

```bash
git clone https://github.com/c0zaut/rkllm-gradio && cd rkllm-gradio
```

You can either setup the virtual environment yourself, or run the setup script:

```bash 
bash setup.sh
```

And enter the virtual environment, with all dependencies installed:

```bash
source ~/.venv/rkllm-gradio/bin/activate
(rkllm_gradio) you@hostname: ~/rkllm-gradio $ 
```

If you setup the virtual environment yourself, you can use the provided requirements.txt file for quick dependency resolution.

```bash
(rkllm_gradio) you@hostname: ~/rkllm-gradio $ python -m pip install --upgrade -r requirements.txt
```

Once the application is setup, you will need to download and setup the models.

- Head over to https://huggingface.co/c01zaut and start downloading models!
- Copy the downloaded models to this repo's `./models` directory
- Update the `model_configs` dictionary in `model_configs.py` with the correct filename of the model, and update any parameters as you see fit

With models in place, and `available_models` containing at least 1 local model, you can start the app with:

```bash
(rkllm_gradio) you@hostname: ~/rkllm-gradio $ python rkllm_server_gradio.py
```

Then head over to localhost:8080 in your browser:

![browser](assets/browser.png)

Select your model:

![model-select](assets/select-model.png)
![model-selected](assets/selected-model.png)

And chat:

![chat](assets/chat.png)

## Default Version

The default version of the RKLLM library, in `./lib/` is 1.1.2. To change to 1.1.1:

```bash
(rkllm_gradio) you@hostname: ~/rkllm-gradio $ cp -p ./lib/librkllmrt.so.111  ./lib/./lib/librkllmrt.so
```

To change back to 1.1.2:

```bash
(rkllm_gradio) you@hostname: ~/rkllm-gradio $ cp -p ./lib/librkllmrt.so.112  ./lib/./lib/librkllmrt.so
```

## Features

- Chat template is auto generated with Transformers! No more setting "PREFIX" and "POSTFIX" manually!
- Customizable parameters for each model family, including system prompt
- txt2txt LLM inference, accelerated by the RK3588 NPU in a single, easy-to-use interface
- Tabs for selecting model, txt2txt (chat,) and txt2mesh (Llama 3.1 8B finetune.)
- txt2mesh: generate meshes with an LLM! **Needs work - large amount of accuracy loss**

![chair](assets/chair-w8a8_g512-opt-0-hybrid-1.0.png)
![sword](assets/sword-w8a8_g512-opt-0-hybrid-1.0.png)
![pyramid](assets/pyramid-w8a8_g512-opt-0-hybrid-1.0.png)

## Limitations

- I get matmul errors when using contexts that are larger than 4096. This occurs on both 1.1.1 and 1.1.2. Inference still completes, and I have gotten coherent output.

```
* Running on local URL:  http://0.0.0.0:8080

To create a public link, set `share=True` in `launch()`.
No model loaded! Continuing with initialization...
=========INITIALIZING===========
I rkllm: rkllm-runtime version: 1.1.2, rknpu driver version: 0.9.7, platform: RK3588

RKLLM Model, internlm2_5-1_8b-chat-w8a8_g512-opt has been initialized successfully！
==============================

E RKNN: [00:45:12.110] meet unkown shape, op name: matmul_qkv_rkllm_spilt_1, shape: 64, 4160, 128
2features matmul matmul run failed
E RKNN: [00:45:12.110] meet unkown shape, op name: matmul_qkv_rkllm_spilt_2, shape: 64, 4160, 128
2features matmul matmul run failed
E RKNN: [00:45:12.125] meet unkown shape, op name: matmul_qk_rkllm_spilt_2, shape: 64, 128, 4160
2features matmul matmul run failed
E RKNN: [00:45:12.125] meet unkown shape, op name: matmul_qk_rkllm_spilt_1, shape: 64, 128, 4160

...

E RKNN: [00:45:13.315] meet unkown shape, op name: matmul_qk_rkllm_spilt_0, shape: 64, 128, 4224
2features matmul matmul run failed
E RKNN: [00:45:13.321] meet unkown shape, op name: matmul_qkv_rkllm_spilt_0, shape: 64, 4224, 128
E RKNN: [00:45:13.321] meet unkown shape, op name: matmul_qkv_rkllm_spilt_1, shape: 64, 4224, 128
2features matmul matmul run failed
2features matmul matmul run failed

...

E RKNN: [00:45:13.546] meet unkown shape, op name: matmul_qk_rkllm_spilt_0, shape: 64, 128, 4288
2features matmul matmul run failed
E RKNN: [00:45:13.553] meet unkown shape, op name: matmul_qkv_rkllm_spilt_1, shape: 64, 4288, 128
E RKNN: [00:45:13.553] meet unkown shape, op name: matmul_qkv_rkllm_spilt_2, shape: 64, 4288, 128
2features matmul matmul run failed
2features matmul matmul run failed

...

--------------------------------------------------------------------------------------
 Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second       
--------------------------------------------------------------------------------------
 Prefill       48433.63         5052      9.59                     104.31                  
 Generate      3751388.33       8191      458.65                   2.18                    
--------------------------------------------------------------------------------------
```

- This is not a production-ready application. It cannot properly handle concurrency, or if users on the same network attempt to do things like load a model at the same time, or attempt to query the model simultaneously. 

- As of this time, only txt2txt models without LoRAs or prompt caches are supported.

- Some of the settings like top_k, top_p, and temperature have to manually adjusted inside of the `model_class.py` script.

## TO DO:

- Add support for multi-modal models

- Incorporate Stable Diffusion: https://huggingface.co/happyme531/Stable-Diffusion-1.5-LCM-ONNX-RKNN2

- Change model dropdown to radio buttons

- Include text box input for system prompt

- Support prompt cache

- Add monitoring for system resources, such as NPU, CPU, GPU, and RAM