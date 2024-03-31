import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import torch
from torch._dynamo import config, explain
from llama_index.core import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks



def get_vectorstore(pdf_docs):
    from langchain_community.document_loaders import TextLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from io import StringIO
    from langchain_community.document_loaders import PyPDFLoader

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents=[]
    for file in pdf_docs:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        docs = text_splitter.split_documents(pages)
        documents.extend(docs)

    from langchain.embeddings import SentenceTransformerEmbeddings
    embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    from langchain.vectorstores import Chroma
    db = Chroma.from_documents(documents, embeddings)
    return db



def get_conversation_chain(vectorstore):
    from langchain.prompts import PromptTemplate
    #llm = ChatOpenAI()
    id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    id_main="mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceHub(repo_id=id_main,
    model_kwargs={"temperature":0.5, "max_length":512,"max_new_tokens":4096,"max_input_length":4096})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
   

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True,
        return_source_documents=False,
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    import re
    ext = re.compile(r'Helpful Answer:')
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # st.write(type(message.content))
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            for line in message.content.split('\n'):
                x1=re.search(ext,line)
                if x1:
                    start=x1.span()[0]
                    end=x1.span()[1]
                    key=line[start:end]
                    value=line[end:]
            st.write(bot_template.replace(
                "{{MSG}}",value), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        files = []
        if pdf_docs:
            for idx, file in enumerate(pdf_docs):
                with open("uploads/"+ str(idx) + ".pdf", "wb") as f:
                    f.write(file.getbuffer())
                files.append("uploads/"+ str(idx) + ".pdf")
        st.write(files)

        if st.button("Process"):
            with st.spinner("Processing"):


                # create vector store
                vectorstore = get_vectorstore(files)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()