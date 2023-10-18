from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env into os.environ

openAI_key = os.environ.get('openAI_key')


llm = OpenAI(openai_api_key=openAI_key)

print('hello main')
print('loading...')
result = llm('Write a poem about the sun')
print(result)
