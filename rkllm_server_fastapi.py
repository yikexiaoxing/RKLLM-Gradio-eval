from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone
import time

# Assuming RKLLMLoaderClass and other necessary imports are available
from ctypes_bindings import *
from model_class import *

# Set resource limit
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

# Initialize FastAPI app
app = FastAPI()

# Global variable for loaded model
rkllm_model = None

# Define the input model for the generate request
class GenerateRequest(BaseModel):
    model: str  # Required model name
    prompt: str  # Required prompt text
    history: Optional[List[str]] = None  # History of messages

# Define the response model
class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: List[str]
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

# Define the input model for the inference request
class InferenceRequest(BaseModel):
    message: str
    history: List[str]  # History of messages

# Helper function to initialize RKLLM model
def initialize_model(model_name: str):
    global rkllm_model
    if rkllm_model is not None:
        return  # The model is already loaded, no need to load it again.

    try:
        # Initialize RKLLM model
        rkllm_model = RKLLMLoaderClass(model=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing model: {str(e)}")

# Helper function to get and process LLM output
def get_RKLLM_output(message, history):
    try:
        # Get the generator object from the model
        output_gen = rkllm_model.get_RKLLM_output(message, history)

        # Initialize a variable to hold the output
        last_content = None

        for item in output_gen:
            # Check if the item is a dictionary with 'content' key
            if isinstance(item, dict) and 'content' in item:
                last_content = item['content']  # Store the last content as we go

            # If the item doesn't match the expected type, we handle it here
            elif isinstance(item, str):
                last_content = item  # In case it's just a string, we use it as the final output

        if last_content:
            return last_content  # Return the last content (final part of the output)
        else:
            return "Empty response from model!"  # Fallback if no output is collected

    except RuntimeError as e:
        return f"Error: {str(e)}"

# API Endpoint: /api/tags (Lists available models)
@app.get("/api/tags")
async def list_models():
    try:
        models = available_models()  # Call the function to get available models
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

# API Endpoint for /api/initialize_model (To initialize the model once)
@app.post("/api/initialize_model/")
async def initialize_model_api(model_name: str):
    try:
        initialize_model(model_name)
        return {"status": "success", "model_name": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing model: {str(e)}")

# API endpoint for /api/generate
@app.post("/api/generate")
async def generate(request: GenerateRequest):
    try:
        # Initialize the model if it's not already loaded
        if rkllm_model is None:
            initialize_model(request.model)

        # History of context messages (default to empty list if not provided)
        context = request.history if request.history else []

        # Generate the response based on the model and request data
        response_content = get_RKLLM_output(request.prompt, context)

        return {
            "model": request.model,
            "created_at": datetime.now(timezone.utc).isoformat(),  # Updated to ISO 8601 format
            "response": response_content,
            "done": True,
            "context": context,  # Send the dynamic context back
            "total_duration": 27075377941,  # Placeholder for actual duration calculation
            "load_duration": 0,  # Placeholder for load duration
            "prompt_eval_count": 0,  # Placeholder for evaluation count
            "prompt_eval_duration": 0,  # Placeholder for evaluation duration
            "eval_count": 0,  # Placeholder for evaluation count
            "eval_duration": 0  # Placeholder for evaluation duration
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Inference API endpoint (already present)
@app.post("/inference/")
async def inference_api(request: InferenceRequest):
    message = request.message
    history = request.history

    # Collect the output from get_RKLLM_output and return as response
    response = get_RKLLM_output(message, history)

    return {"response": response}

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run FastAPI server

