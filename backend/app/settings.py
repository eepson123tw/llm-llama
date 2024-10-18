import os
from typing import Dict

from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


apikey = os.getenv("OPENAI_API_KEY")

print(apikey)

def llm_config_from_env() -> Dict:
    from llama_index.core.constants import DEFAULT_TEMPERATURE

    model = os.getenv("MODEL", "gpt-4-turbo-preview")
    temperature = os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    max_tokens = os.getenv("LLM_MAX_TOKENS")

    # 防護處理 max_tokens 是否為 None
    config = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
        "api_key": apikey
    }
    return config


def embedding_config_from_env() -> Dict:
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    dimension = os.getenv("EMBEDDING_DIM")

    # 防護處理 dimension 是否為 None
    config = {
        "model": model,
        "dimension": int(dimension) if dimension is not None else None,
        "api_key": apikey
    }
    return config


def init_settings():
    llm_configs = llm_config_from_env()
    embedding_configs = embedding_config_from_env()

    # 設置 LLM 和 Embedding 模型配置
    Settings.llm = OpenAI(**llm_configs)
    Settings.embed_model = OpenAIEmbedding(**embedding_configs)

    # 設置 chunk 的大小和重疊度，使用環境變量或默認值
    Settings.chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
    Settings.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "20"))

    # 初始化回調管理器，啟用 LlamaDebugHandler 進行跟蹤
    Settings.callback_manager = CallbackManager([LlamaDebugHandler(print_trace_on_end=True)])

