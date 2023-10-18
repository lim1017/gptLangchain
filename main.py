from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

import argparse
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env into os.environ

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='return a list of numbers')
parser.add_argument('--language', type=str, default='python')
args = parser.parse_args()

openAI_key = os.environ.get('openAI_key')


llm = OpenAI(openai_api_key=openAI_key)

print('hello')

code_prompt = PromptTemplate(
    template='Write a short {language} function that will perform {task}',
    input_variables=['language', 'task']
)

print(code_prompt.template)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key='code')

code_review_prompt = PromptTemplate(
    template='Please review this {code}, and make sure its free of bugs, then write a test for it',
    input_variables=['code']
)

code_review_chain = LLMChain(llm=llm, prompt=code_review_prompt)

chain = SequentialChain(
    chains=[code_chain, code_review_chain],
    input_variables=['task', 'language'],
    output_variables=['code', 'text'],
)

resultDict = chain({"language": args.language, "task": args.task})

resultCode = resultDict['code']
resultText = resultDict['text']

print('----CODE----')
print(resultCode)
print('----TEXT----')
print(resultText)
