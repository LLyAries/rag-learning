import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# 从项目根目录加载 .env
load_dotenv(ENV_PATH)


def create_llm():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL")
    model_name = os.getenv("LLM_MODEL")

    if not api_key:
        raise ValueError(
            ".env 中缺少 DASHSCOPE_API_KEY"
        )

    if not base_url:
        raise ValueError(
            ".env 中缺少 DASHSCOPE_BASE_URL"
        )

    if not model_name:
        raise ValueError(
            ".env 中缺少 LLM_MODEL"
        )

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0,
    )

    return llm


if __name__ == "__main__":
    llm = create_llm()

    response = llm.invoke(
        "请用一句中文解释什么是 RAG。"
    )

    print("模型回答：")
    print(response.content)