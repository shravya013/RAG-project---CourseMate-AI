from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document loaders/work paper.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)



# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.prompts import ChatPromptTemplate
# load_dotenv()

# data = PyPDFLoader("document loaders/work paper.txt")
# docs = data.load()

# template = ChatPromptTemplate.from_messages(
#     [("system","you are a ai that summarizes the text"),
#      ("human","{data}")]
#     )

# model = ChatGroq(model="openai/gpt-oss-20b")

# prompt = template.format_messages(data = docs[0].page_content)

# result = model.invoke(prompt)

# print(result.content)