import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 RAG Document Assistant")

st.write(
    "Upload a PDF, create the vector database, "
    "and ask questions about the document."
)


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


# --------------------------------------------------
# Prompt template
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Use ONLY the provided context to answer the question.

        If the answer is not present in the context,
        say:

        "I could not find the answer in the document."

        Do not use outside knowledge.
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])


# --------------------------------------------------
# PDF upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# Create vector database
# --------------------------------------------------

if uploaded_file is not None:

    st.write(
        f"Selected file: **{uploaded_file.name}**"
    )

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            # ------------------------------------------
            # Read PDF
            # ------------------------------------------

            reader = PdfReader(uploaded_file)

            docs = []

            for page_number, page in enumerate(reader.pages):

                text = page.extract_text()

                if text:

                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "page": page_number + 1,
                                "source": uploaded_file.name
                            }
                        )
                    )


            # ------------------------------------------
            # Check if text was extracted
            # ------------------------------------------

            if not docs:

                st.error(
                    "Could not extract text from this PDF."
                )

                st.stop()


            # ------------------------------------------
            # Split document into chunks
            # ------------------------------------------

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)


            # ------------------------------------------
            # Create persistent Chroma database
            # ------------------------------------------

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma_db"
            )


            # ------------------------------------------
            # Create retriever
            # ------------------------------------------

            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )


            # ------------------------------------------
            # Store retriever in Streamlit session
            # ------------------------------------------

            st.session_state.retriever = retriever
            st.session_state.document_name = uploaded_file.name
            st.session_state.database_created = True


        st.success(
            "Vector database created successfully!"
        )


# --------------------------------------------------
# Question-answering section
# --------------------------------------------------

if st.session_state.get("database_created", False):

    st.subheader("💬 Ask a question")

    query = st.text_input(
        "Enter your question"
    )


    if st.button("Ask"):

        if not query.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner("Searching the document..."):

                # --------------------------------------
                # Retrieve relevant documents
                # --------------------------------------

                docs = st.session_state.retriever.invoke(
                    query
                )


                # --------------------------------------
                # Create context
                # --------------------------------------

                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )


                # --------------------------------------
                # Create final prompt
                # --------------------------------------

                final_prompt = prompt.invoke({
                    "context": context,
                    "question": query
                })


                # --------------------------------------
                # Get answer from Groq
                # --------------------------------------

                response = llm.invoke(
                    final_prompt
                )


            # ------------------------------------------
            # Display answer
            # ------------------------------------------

            st.subheader("🤖 Answer")

            st.write(response.content)


            # ------------------------------------------
            # Display retrieved chunks
            # ------------------------------------------

            with st.expander(
                "🔎 View Retrieved Document Chunks"
            ):

                for i, doc in enumerate(docs):

                    st.markdown(
                        f"### Chunk {i + 1}"
                    )

                    st.write(
                        doc.page_content
                    )

                    st.caption(
                        f"Page: {doc.metadata.get('page', 'Unknown')}"
                    )

                    st.divider()