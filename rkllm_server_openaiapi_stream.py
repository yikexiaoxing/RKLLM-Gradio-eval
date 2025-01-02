from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncGenerator, Union
import time
import os
import resource
import random
import string
import logging
import json
import asyncio
from model_class import available_models, RKLLMLoaderClass
from model_configs import model_configs
from transformers import AutoTokenizer

resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))
app = FastAPI()
rkllm_model = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Middleware to log incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Headers: {request.headers}")
    try:
        body = await request.json()
        logger.info(f"Body: {body}")
    except Exception:
        logger.info("Body: <Not JSON>")
    response = await call_next(request)
    return response


async def stream_tokens(content: str) -> AsyncGenerator[str, None]:
    """Split content into tokens and stream them with simulated delay."""
    tokens = content.split()
    for token in tokens:
        yield f"data: {json.dumps({'choices': [{'delta': {'content': token + ' '}}]})}\n\n"
        await asyncio.sleep(0.05)  # Simulate natural typing speed
    yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


# Define the input model for the completion request

class Message(BaseModel):
    role: str  # one of: "system", "user", "assistant"
    content: str

class CompletionRequest(BaseModel):
    model: str  # e.g., "text-davinci-003", "gpt-3.5-turbo", etc.
    prompt: Union[str, List[str]]  # Single string or a list of strings
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)  # Controls randomness (0.0 - 1.0)
    max_tokens: Optional[int] = None  # Max number of tokens for the response
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)  # Top-p (nucleus sampling)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)  # Penalize new tokens based on frequency
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)  # Penalize based on token presence
    stop: Optional[Union[str, List[str]]] = None  # Stop sequences (can be a list or a single string)



# Define the input model for the chat completion request
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None





# Define the response model for completions
class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict


# Define the response model for chat completions
class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict


# Helper function to generate a dynamic completion ID
def generate_completion_id():
    # Generate a random string of 24 characters (letters and digits)
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=24))
    # Prefix with "cmpl-" to match OpenAI's format
    return f"cmpl-{random_string}"


# Helper function to initialize RKLLM model
def initialize_model(model_name: str):
    global rkllm_model
    if rkllm_model is not None:
        return  # The model is already loaded, no need to load it again.

    try:
        # Initialize RKLLM model
        rkllm_model = RKLLMLoaderClass(model=model_name)
        logger.info(f"Model '{model_name}' initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing model '{model_name}': {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error initializing model: {str(e)}"
        )


# Helper function to count tokens using AutoTokenizer
def count_tokens(text: str, model_name: str) -> int:
    try:
        models = available_models()
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found.")
        st_model_id = models[model_name]["config"]["st_model_id"]
        tokenizer = AutoTokenizer.from_pretrained(st_model_id, trust_remote_code=True)
        tokens = tokenizer.tokenize(text)
        return len(tokens)
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


# Helper function to get and process LLM output
def get_RKLLM_output(
    message,
    history=None,
    max_tokens=None,
    temperature=None,
    top_p=None,
    frequency_penalty=None,
    presence_penalty=None,
):
    try:
        # Ensure max_tokens is within the model's limit
        if max_tokens > 2048:  # Adjust this limit as needed
            max_tokens = 2048

        # Get the generator object from the model
        output_gen = rkllm_model.get_RKLLM_output(
            message, history or [], max_tokens=max_tokens
        )

        # Initialize a variable to hold the output
        last_content = []

        for item in output_gen:
            last_content.append(
                item
            )  # In case it's just a string, we use it as the final output

        if last_content:
            return last_content  # Return the last content (final part of the output)
        else:
            return "Empty response from model!"  # Fallback if no output is collected

    except RuntimeError as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}"


async def get_RKLLM_output_stream(
    message,
    history=None,
    max_tokens=None,
    temperature=None,
    top_p=None,
    frequency_penalty=None,
    presence_penalty=None,
):
    try:
        if max_tokens > 2048:
            max_tokens = 2048

        output_gen = rkllm_model.get_RKLLM_output(
            message, history or [], max_tokens=max_tokens
        )

        for item in output_gen:
            if isinstance(item, dict) and "content" in item:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': item['content']}}]})}\n\n"
            elif isinstance(item, str):
                yield f"data: {json.dumps({'choices': [{'delta': {'content': item}}]})}\n\n"
        yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    except RuntimeError as e:
        logger.error(f"Error generating response: {str(e)}")
        raise StopAsyncIteration(f"Error: {str(e)}")


# API Endpoint: /v1/models (Lists available models)
@app.get("/v1/models")
async def list_models():
    try:
        models = available_models()
        formatted_models = []
        for model, details in models.items():
            model_path = os.path.join("./models", details["filename"])
            if os.path.exists(model_path):
                # Get the file's last modification time
                file_timestamp = os.path.getmtime(model_path)
                # Convert to an integer (Unix timestamp)
                created_timestamp = int(file_timestamp)
            else:
                # Fallback to the current time if the file doesn't exist
                created_timestamp = int(time.time())

            formatted_models.append(
                {
                    "id": model,  # Use the model name as the ID
                    "object": "model",
                    "created": created_timestamp,  # Use the file's modification time
                    "owned_by": "rkllm",  # Set the owner to "rkllm" or your organization
                }
            )

        return {"object": "list", "data": formatted_models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


# API Endpoint: /v1/completions (Generates text completions)
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    try:
        if request.stream:
            # Validate the request
            if not request.model:
                raise HTTPException(status_code=400, detail="Model name is required")
            if not request.prompt:
                raise HTTPException(status_code=400, detail="Prompt is required")

            # Initialize the model if it's not already loaded
            if rkllm_model is None:
                initialize_model(request.model)

            # Get the model's default configuration from model_configs
            models = available_models()
            if request.model not in models:
                raise HTTPException(
                    status_code=404, detail=f"Model '{request.model}' not found"
                )

            model_details = models[request.model]
            family = model_details["family"]
            base_config = model_configs[family]["base_config"]

            # Cap max_tokens to the model's max_new_tokens limit and remaining context length
            max_tokens = (
                min(request.max_tokens, base_config["max_new_tokens"])
                if request.max_tokens is not None
                else base_config["max_new_tokens"]
            )
            prompt_tokens = count_tokens(request.prompt, request.model)
            remaining_tokens = base_config["max_context_len"] - prompt_tokens
            max_tokens = min(max_tokens, remaining_tokens)
            logger.info(f"Using max_tokens: {max_tokens}")

            # Check if the prompt exceeds the model's context length
            if prompt_tokens > base_config["max_context_len"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Token count ({prompt_tokens}) exceeds the model's max_context_len ({base_config['max_context_len']}).",
                )

            # Use the request parameters if provided, otherwise fall back to the model's default configuration
            temperature = (
                request.temperature
                if request.temperature is not None
                else base_config["temperature"]
            )
            top_p = request.top_p if request.top_p is not None else base_config["top_p"]
            frequency_penalty = (
                request.frequency_penalty
                if request.frequency_penalty is not None
                else base_config["frequency_penalty"]
            )
            presence_penalty = (
                request.presence_penalty
                if request.presence_penalty is not None
                else base_config.get("presence_penalty", 0.0)
            )

            # Generate the response based on the model and request data
            return StreamingResponse(
                get_RKLLM_output_stream(
                    message=request.prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                ),
                media_type="text/event-stream",
            )
            logger.info(f"Response: {response}")

        else:
            # Validate the request
            if not request.model:
                raise HTTPException(status_code=400, detail="Model name is required")
            if not request.prompt:
                raise HTTPException(status_code=400, detail="Prompt is required")

            # Initialize the model if it's not already loaded
            if rkllm_model is None:
                initialize_model(request.model)

            # Get the model's default configuration from model_configs
            models = available_models()
            if request.model not in models:
                raise HTTPException(
                    status_code=404, detail=f"Model '{request.model}' not found"
                )

            model_details = models[request.model]
            family = model_details["family"]
            base_config = model_configs[family]["base_config"]

            # Cap max_tokens to the model's max_new_tokens limit and remaining context length
            max_tokens = (
                min(request.max_tokens, base_config["max_new_tokens"])
                if request.max_tokens is not None
                else base_config["max_new_tokens"]
            )
            prompt_tokens = count_tokens(request.prompt, request.model)
            remaining_tokens = base_config["max_context_len"] - prompt_tokens
            max_tokens = min(max_tokens, remaining_tokens)
            logger.info(f"Using max_tokens: {max_tokens}")

            # Check if the prompt exceeds the model's context length
            if prompt_tokens > base_config["max_context_len"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Token count ({prompt_tokens}) exceeds the model's max_context_len ({base_config['max_context_len']}).",
                )

            # Use the request parameters if provided, otherwise fall back to the model's default configuration
            temperature = (
                request.temperature
                if request.temperature is not None
                else base_config["temperature"]
            )
            top_p = request.top_p if request.top_p is not None else base_config["top_p"]
            frequency_penalty = (
                request.frequency_penalty
                if request.frequency_penalty is not None
                else base_config["frequency_penalty"]
            )
            presence_penalty = (
                request.presence_penalty
                if request.presence_penalty is not None
                else base_config.get("presence_penalty", 0.0)
            )

            # Generate the response based on the model and request data
            response_content = get_RKLLM_output(
                message=request.prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            # Count tokens in the response
            completion_tokens = count_tokens(response_content, request.model)
            logger.info(f"Token count in response: {completion_tokens}")

            # Determine the finish reason
            finish_reason = "stop" if completion_tokens < max_tokens else "length"
            if finish_reason == "length":
                logger.warning(
                    f"Response truncated due to token limit. Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Max tokens: {max_tokens}"
                )

            # Generate a dynamic completion ID
            completion_id = generate_completion_id()

            # Log the full response
            response = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "text": response_content,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            logger.info(f"Response: {response}")

            return response
    except HTTPException as e:
        logger.error(f"HTTPException: {str(e.detail)}")
        raise e
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )


# API Endpoint: /v1/chat/completions (Generates chat completions)
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        if request.stream:
            # Validate the request
            if not request.model:
                raise HTTPException(status_code=400, detail="Model name is required")
            if not request.messages:
                raise HTTPException(status_code=400, detail="Messages are required")

            # Initialize the model if it's not already loaded
            if rkllm_model is None:
                initialize_model(request.model)

            # Get the model's default configuration from model_configs
            models = available_models()
            if request.model not in models:
                raise HTTPException(
                    status_code=404, detail=f"Model '{request.model}' not found"
                )

            model_details = models[request.model]
            family = model_details["family"]
            base_config = model_configs[family]["base_config"]

            # Cap max_tokens to the model's max_new_tokens limit and remaining context length
            max_tokens = (
                min(request.max_tokens, base_config["max_new_tokens"])
                if request.max_tokens is not None
                else base_config["max_new_tokens"]
            )
            prompt = "\n".join([msg["content"] for msg in request.messages])
            prompt_tokens = count_tokens(prompt, request.model)
            remaining_tokens = base_config["max_context_len"] - prompt_tokens
            max_tokens = min(max_tokens, remaining_tokens)
            logger.info(f"Using max_tokens: {max_tokens}")

            # Check if the prompt exceeds the model's context length
            if prompt_tokens > base_config["max_context_len"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Token count ({prompt_tokens}) exceeds the model's max_context_len ({base_config['max_context_len']}).",
                )

            # Use the request parameters if provided, otherwise fall back to the model's default configuration
            temperature = (
                request.temperature
                if request.temperature is not None
                else base_config["temperature"]
            )
            top_p = request.top_p if request.top_p is not None else base_config["top_p"]
            frequency_penalty = (
                request.frequency_penalty
                if request.frequency_penalty is not None
                else base_config["frequency_penalty"]
            )
            presence_penalty = (
                request.presence_penalty
                if request.presence_penalty is not None
                else base_config.get("presence_penalty", 0.0)
            )

            prompt = "\n".join([msg["content"] for msg in request.messages])
            return StreamingResponse(
                get_RKLLM_output_stream(
                    message=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                ),
                media_type="text/event-stream",
            )
        else:
            # Validate the request
            if not request.model:
                raise HTTPException(status_code=400, detail="Model name is required")
            if not request.messages:
                raise HTTPException(status_code=400, detail="Messages are required")

            # Initialize the model if it's not already loaded
            if rkllm_model is None:
                initialize_model(request.model)

            # Get the model's default configuration from model_configs
            models = available_models()
            if request.model not in models:
                raise HTTPException(
                    status_code=404, detail=f"Model '{request.model}' not found"
                )

            model_details = models[request.model]
            family = model_details["family"]
            base_config = model_configs[family]["base_config"]

            # Cap max_tokens to the model's max_new_tokens limit and remaining context length
            max_tokens = (
                min(request.max_tokens, base_config["max_new_tokens"])
                if request.max_tokens is not None
                else base_config["max_new_tokens"]
            )
            prompt = "\n".join([msg["content"] for msg in request.messages])
            prompt_tokens = count_tokens(prompt, request.model)
            remaining_tokens = base_config["max_context_len"] - prompt_tokens
            max_tokens = min(max_tokens, remaining_tokens)
            logger.info(f"Using max_tokens: {max_tokens}")

            # Check if the prompt exceeds the model's context length
            if prompt_tokens > base_config["max_context_len"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Token count ({prompt_tokens}) exceeds the model's max_context_len ({base_config['max_context_len']}).",
                )

            # Use the request parameters if provided, otherwise fall back to the model's default configuration
            temperature = (
                request.temperature
                if request.temperature is not None
                else base_config["temperature"]
            )
            top_p = request.top_p if request.top_p is not None else base_config["top_p"]
            frequency_penalty = (
                request.frequency_penalty
                if request.frequency_penalty is not None
                else base_config["frequency_penalty"]
            )
            presence_penalty = (
                request.presence_penalty
                if request.presence_penalty is not None
                else base_config.get("presence_penalty", 0.0)
            )

            # Generate the response based on the model and request data
            response_content = get_RKLLM_output(
                message=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            # Count tokens in the response
            completion_tokens = len(response_content)

            response_content = "".join(response_content)

            logger.info(f"Token count in response: {completion_tokens}")

            # Determine the finish reason
            finish_reason = "stop" if completion_tokens < max_tokens else "length"
            if finish_reason == "length":
                logger.warning(
                    f"Response truncated due to token limit. Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Max tokens: {max_tokens}"
                )

            # Generate a dynamic completion ID
            completion_id = generate_completion_id()

            # Log the full response
            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": response_content},
                        "index": 0,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            logger.info(f"Response: {response}")

            return response
    except HTTPException as e:
        logger.error(f"HTTPException: {str(e.detail)}")
        raise e
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )


# Run FastAPI server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
