# load pdf
# split into chunks
# store into chromaDB

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document loaders/work paper.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding = embedding_model,
    persist_directory="chroma_db"
)