import io
import os
import torch
from fastapi import APIRouter, FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import logging
import requests
from typing import Any, Dict, List, Optional
import re
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize APIRouter
llama_chat_router = APIRouter()

def clean_output(text):
    # 移除所有 <|...|> 标记
    clean_text = re.sub(r"<\|.*?\|>", "", text)
    return clean_text.strip()

# Healthcheck endpoint
@llama_chat_router.get("/healthcheck")
def healthcheck():
    print('123123')
    return JSONResponse(content={"response": 'Hello world!'})


# Load model and processor
def load_model():
    global model, processor
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    try:
        # Get Hugging Face Token from environment variables
        hubToken = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hubToken:
            raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in environment variables.")
        
        login(token=hubToken)
        
        # Load model with bfloat16
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_id)
        
        logger.info("Model and processor loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e


@llama_chat_router.get("/check_gpu_memory")
def check_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{i}")
            total_mem = torch.cuda.get_device_properties(device).total_memory
            reserved_mem = torch.cuda.memory_reserved(device)
            allocated_mem = torch.cuda.memory_allocated(device)
            free_mem = reserved_mem - allocated_mem
            print(f"GPU {i}:")
            print(f"  总内存: {total_mem / (1024 ** 3):.2f} GB")
            print(f"  已分配内存: {allocated_mem / (1024 ** 3):.2f} GB")
            print(f"  保留内存: {reserved_mem / (1024 ** 3):.2f} GB")
            print(f"  空闲内存: {free_mem / (1024 ** 3):.2f} GB")
    else:
        print("CUDA 不可用。")


# Llama generation endpoint
@llama_chat_router.post("/generate_llama")
def generate_llama():
    try:
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)


        output = model.generate(**inputs, max_new_tokens=30)
        decoded_output = processor.decode(output[0])

        clean_text = clean_output(decoded_output)

        return JSONResponse(content={"response": clean_text})
    except Exception as e:
        logger.error(f"Error in generate_llama: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


