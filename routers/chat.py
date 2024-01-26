from typing import Any,AsyncIterable
from fastapi import APIRouter,Depends, File, Form, UploadFile
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import milvus

from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
# from langchain.memory import ChatMessageHistory
# from langchain.chains import LLMChain
from langchain.chains import ConversationChain, ConversationalRetrievalChain, LLMChain

# from .langchain.sayHello import sayHello
from PyPDF2 import PdfReader
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks import AsyncIteratorCallbackHandler

from utils.index import generate_random_string

import asyncio

from functools import lru_cache
import config


router = APIRouter()

chat_history = ChatMessageHistory()


class ChatQueryModel(BaseModel):
    query: str
    collectionName: str


class CallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self) -> None:
        self.content:str=''
        self.final_answer:bool = False
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content+=token
        # if 'Final Answer' in self.content:
        #     self.final_answer = True
        # if self.final_answer:
        #     if '"action_input:"' in self.content:
        #         if token not in ["}"]:
        #             sys.stdout.write(token)
        #             sys.stdout.flush()

@lru_cache()
def get_settings():
    return config.Settings()
# settings = get_settings()()
# print('settings',settings)
# settings = config.settings

@router.post("/pdfToEmbeddings")
async def root(file: UploadFile = File(...)):
    # print()
    settings = config.settings
    print('settings',settings)
    
    psf_reader = PdfReader(file.file)
    print('psf_reader',psf_reader)
    text = ""
    for page in psf_reader.pages:
        text += page.extract_text()
    text=text.replace("[", "").replace("]", "")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                          chunk_overlap=20,
                                          separators=["\n\n", "\n", " ", ""],
                                          length_function=len)
    docs = text_splitter.split_text(text)
    print('length',len(docs))
    embeddings = OpenAIEmbeddings(base_url=settings.openai_api_base, api_key=settings.openai_api_key)
    collectionName ='langchain_collection_'+generate_random_string(5)
    vector_db = milvus.Milvus.from_texts(
        docs,
        embeddings,
        connection_args={
            # 'uri':milvus_url,
            'host': settings.milvus_host,
            'port': settings.milvus_port,
            'user': settings.milvus_username,
            'password': settings.milvus_password,
            'secure': False
        },
        collection_name=collectionName
    )
    return {"collectionName": collectionName}



async def send_message(chatQuery: ChatQueryModel)->AsyncIterable[str]:
    settings = config.settings
    print('settings',config.settings)
    query = chatQuery.query
    collectionName = chatQuery.collectionName
    embeddings = OpenAIEmbeddings(base_url=settings.openai_api_base, api_key=settings.openai_api_key)
    vectorSource = milvus.Milvus(
        embeddings,
        connection_args={
            "host": settings.milvus_host,
            "port": settings.milvus_port
        },
        collection_name=collectionName,
    )
    # queryResult = vectorSource.similarity_search(query)

    # memory = ConversationBufferMemory(memory_key="chat_history")
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,chat_memory=chat_history)
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)
    
    callback = AsyncIteratorCallbackHandler()
    streaming_llm = ChatOpenAI(temperature=0,streaming=True,callbacks=[callback])
    qa = ConversationalRetrievalChain.from_llm(
        streaming_llm,
        retriever=vectorSource.as_retriever(),
        # callbacks=[callback],
        memory=memory)
    task = asyncio.create_task(
        qa.arun({'question': query})
    )
    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f" Caught exception:{e}")
    finally:
        callback.done.set()
    await task


#根据文章内容对话
@router.post("/queryDocuments")
async def root(chatQuery: ChatQueryModel):
    generator = send_message(chatQuery)
    return StreamingResponse(generator, media_type="text/event-stream")

# 生成文章摘要
@router.post("/getSummary")
async def root(file: UploadFile = File(...)):
    return {"message": "Hello World"}
