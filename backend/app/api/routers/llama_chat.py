import io
import os
import json
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

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel



openai_api_base = "http://localhost:8000/v1"


llm = OpenAILike(api_base=openai_api_base, api_key="fake", model="/app/model/Llama-3.2-11B-Vision-Instruct")


# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )
# llm = VllmServer(
#     api_url="http://localhost:8000/v1", 
#     temperature=0.7
# )


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


@llama_chat_router.post("/generate_llama")
def generate_llama(request: ImageRequest):
    try:
        logger.info("Received request to generate_llama")
        
        # Retrieve the last message safely
        if not request.messages:
            raise ValueError("No messages provided")
        
        last_message = request.messages[-1]  # More Pythonic way to get the last item

        if not last_message.content:
            raise ValueError("No content provided in the last message")
        
        # Initialize variables
        image_url = request.data.imageUrl != "" and request.data.imageUrl or None
        content_list = [{
            "type": "text",
            "text": last_message.content
        }]
        
        # Process image_url if provided
        if image_url:
            match = re.match(r'data:(image/\w+);base64,(.+)', image_url)
            if not match:
                raise ValueError("Invalid imageUrl format")
            
            mime_type, image_base64 = match.groups()
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                },
            }
            content_list.append(image_content)  # Append image content to the content list


        # Construct llama_messages
        llama_messages = [
            {
                "role": "assistant",
                "content":"Description: You are ChatGPT, an AI assistant specialized in analyzing text and images. Tasks: Text - Summarize: Provide a brief summary of main points; Key_elements: List critical components or arguments; Analysis: Offer an objective analysis of messages and implications. Image - Description: Describe visual elements in detail; Key_features: Highlight prominent features or focal points; Interpretation: Analyze possible meanings, themes, or emotions. Guidelines: Clarity: Ensure responses are clear and organized; Conciseness: Keep answers succinct; Objectivity: Maintain an objective perspective; Format: Use bullet points or structured formatting."
            },    
            {
                "role": "user",
                "content": content_list
            }
        ]
        
        logger.debug(f"Llama messages: {llama_messages}")


        messages = [
            ChatMessage(role="system", content="You are CEO of MetaAI"),
            ChatMessage(role="user", content="Introduce Llama3 to the world."),
        ]
        llmRes = llm.chat(messages)

                
        # Create chat completion using the Llama model
        # chat_completion = llm.chat(
        #     messages=llama_messages,
        #     model='/app/model/Llama-3.2-11B-Vision-Instruct',
        # )

        # # Extract the response message
        # message = chat_completion.choices[0].message.content

        # # Prepare the result
        result = {
            "message":  dict(llmRes),
            "url": image_url  # This will be an empty string if no imageUrl was provided
        }

        print(result)
        # Convert the result to JSON
        json_string = json.dumps(result)

        logger.info("Successfully generated response for generate_llama")
        return Response(content=json_string, media_type="application/json")

    except ValueError as ve:
        logger.error(f"Value error in generate_llama: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in generate_llama: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
