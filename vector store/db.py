from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="python is widely used in Aritificial Intelligence",metadata={"source":"AI_book"}),
    Document(page_content="pandas is used for data analysis in python.",metadata={"source":"DataScience_book"}),
    Document(page_content="neural network are used in deep learning.",metadata={"source":"DL_book"})
]

embedding_model = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "chrom-db"
)

result = vector_store.similarity_search("what is used for data analysis?",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)
    
retrieve = vector_store.as_retriever()

docs = retrieve.invoke("explain deep learning")

for d in docs:
    print(d.page_content)