import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# openai.api_key = os.environ.get("OPENAI_API_KEY")
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY') # 设置 OpenAI 的 key
api_base = os.getenv('OPENAI_API_BASE') # 指定代理地址
milvus_host = os.getenv('MILVUS_HOST')
milvus_port = os.getenv('MILVUS_PORT')
milvus_username = os.getenv('MILVUS_USERNAME') 
milvus_password = os.getenv('MILVUS_PASSWORD') 

llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5,openai_api_key=api_key,base_url=api_base)

# en_to_zh_prompt = PromptTemplate(
#     template="请把下面这句话翻译成英文： \n\n {question}?", input_variables=["question"]
# )

en_to_zh_prompt = PromptTemplate.from_template('请把下面这句话翻译成英文： \n\n {question}?')

question_prompt = PromptTemplate(
    template = "{english_question}", input_variables=["english_question"]
)

zh_to_cn_prompt = PromptTemplate(
    input_variables=["english_answer"],
    template="请把下面这一段翻译成中文： \n\n{english_answer}?",
)

question_translate_chain = LLMChain(llm=llm, prompt=en_to_zh_prompt, output_key="english_question")

english = question_translate_chain.run(question="请你作为一个机器学习的专家，介绍一下CNN的原理。")
print(english)

qa_chain = LLMChain(llm=llm, prompt=question_prompt, output_key="english_answer")
english_answer = qa_chain.run(english_question=english)
print(english_answer)

answer_translate_chain = LLMChain(llm=llm, prompt=zh_to_cn_prompt)
answer = answer_translate_chain.run(english_answer=english_answer)
print(answer)