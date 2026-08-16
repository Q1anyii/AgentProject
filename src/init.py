import os

from langchain.chat_models.base import init_chat_model

from langchain_community.document_loaders.text import TextLoader
from langchain_openai import OpenAIEmbeddings
from load_dotenv import load_dotenv

load_dotenv(override=True)

model = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("BASE_URL")
)

embed_model = OpenAIEmbeddings(
 model="BAAI/bge-m3", # 免费模型 ID: BAAI/bge-m3
 base_url=os.getenv("SILICONFLOW_BASE_URL"),
 api_key=os.getenv("SILICONFLOW_API_KEY"),
)

loader = TextLoader("../recourses/prompt/system_prompt.txt", encoding="utf-8")
system_prompt = loader.load()[0].page_content

