import io
import os
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import logging
import requests
from typing import Any, Dict, List, Optional, Union
import re
import base64
import dspy
from openai import OpenAI
from pydantic import BaseModel
# from llama_index.core.llms import ChatMessage
import json

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

llama = dspy.HFClientVLLM(
    model="/app/model/Llama-3.2-11B-Vision-Instruct",
    port=8000,  # port 應為整數
    url="http://localhost",
    seed=42,
    temperature=1  # 設定適當的 temperature 值,
)

dspy.settings.configure(lm=llama)

# 定義 Pydantic 模型
class ImageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ImageContent]]

class Data(BaseModel):
    imageUrl: str

class ImageRequest(BaseModel):
    messages: List[Message]
    data: Data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize APIRouter
llama_chat_router = APIRouter()

def clean_output(text: str) -> str:
    # 移除所有 <|...|> 标记
    clean_text = re.sub(r"<\|.*?\|>", "", text)
    return clean_text.strip()

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Healthcheck endpoint
@llama_chat_router.get("/healthcheck")
def healthcheck():
    return JSONResponse(content={"response": 'Hello world!'})

# Load model and processor
def load_model():
    global model, processor
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    try:
        hub_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hub_token:
            raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in environment variables.")
        
        login(token=hub_token)
        
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
            logger.info(f"GPU {i}:")
            logger.info(f"  Total Memory: {total_mem / (1024 ** 3):.2f} GB")
            logger.info(f"  Allocated Memory: {allocated_mem / (1024 ** 3):.2f} GB")
            logger.info(f"  Reserved Memory: {reserved_mem / (1024 ** 3):.2f} GB")
            logger.info(f"  Free Memory: {free_mem / (1024 ** 3):.2f} GB")
    else:
        logger.warning("CUDA is not available.")

# Llama generation endpoint
@llama_chat_router.post("/generate_llama")
def generate_llama(request: ImageRequest):
    try:
        image_url = request.data.imageUrl
        if not image_url:
            raise ValueError("No imageUrl provided")
        
        match = re.match(r'data:(image/\w+);base64,(.+)', image_url)

        if not match:
            raise ValueError("Invalid imageUrl format")
        
        mime_type, image_base64 = match.groups()

        last_message = request.messages[request.messages.__len__() - 1]

        if not last_message.content:
            raise ValueError("No content provided")
        
   
        classify = dspy.ChainOfThought('question -> answer', n=1)


        response = classify(question=last_message.content)

        print(f"Reasoning: {response.rationale}",last_message.content)

        llama_messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": last_message.content
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    },
                },
            ],
        }]

        chat_completion = client.chat.completions.create(
            messages=llama_messages,
            model='/app/model/Llama-3.2-11B-Vision-Instruct',
            max_tokens=2000,
            
        )

        message = chat_completion.choices[0].message.content
        # Convert the message object to a dictionary
        result = {"message":message,"url":image_url}  # If message is a Pydantic model

        json_string = json.dumps(dict(result))

        return Response(content=json_string, media_type="application/json")

    except Exception as e:
        logger.error(f"Error in generate_llama: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
