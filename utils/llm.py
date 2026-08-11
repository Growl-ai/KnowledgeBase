import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

_llm_client_cache = {}
default_model = os.getenv("LLM_DEFAULT_MODEL")
def get_llm(model: str | None = None, json_mode: bool = False):
    model = model or default_model
    key = (model, json_mode)
    if key in _llm_client_cache:
        return _llm_client_cache[key]

    extra_body = {"enable_thinking": False}
    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    llm = ChatOpenAI(
            model=model,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL", ""),
            extra_body=extra_body,
            model_kwargs=model_kwargs,
        )
    _llm_client_cache[key] = llm
    return llm


if __name__ == '__main__':
    client = get_llm()
    print(client)

    client = get_llm()
    print(client)

    client = get_llm("qwen-max", True)
    print(client)