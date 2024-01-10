from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import milvus
from langchain.document_loaders import TextLoader

def docsToEmbeddings():
    # Load embeddings
    embeddings = OpenAIEmbeddings()

    # Load text
    text_loader = TextLoader()
    text = text_loader.load("data/text.txt")

    splitter = CharacterTextSplitter()
    docs = splitter.split(text)

    # Get embeddings
    embeddings = embeddings.get_embeddings(docs)

    # Save embeddings
    milvus.save_embeddings(embeddings)



