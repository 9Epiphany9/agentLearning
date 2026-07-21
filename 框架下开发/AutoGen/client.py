from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

model_client = OpenAIChatCompletionClient(
    model="deepseek-v4-flash",
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    model_info={
        "function_calling": True,
        "max_tokens": 4096,
        "context_length": 32768,
        "vision": False,
        "json_output": True,
        "family": "deepseek",
        "structured_output": True,
    }
)

