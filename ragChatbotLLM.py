import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import ollama

#page configuration
st.title("C++ RAG Chatbot using LLM")

#load and process data
@st .cache_data
def load_vectorstore():
    loader = TextLoader("CPP.txt", encoding="utf8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_document=text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(final_document, embeddings)
    return db

db = load_vectorstore()

#load LLM(ollama)
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma2:2b")

#chat interface
user_question=st.text_input("Ask a question about C++:")

if user_question:
    with st.spinner("Generating response..."):
        docs=db.similarity_search(user_question)
        context="\n".join([doc.page_content for doc in docs])
        prompt=f"""
        answer the question based on the context below.

        Context: 
        {context}

        Question:
        {user_question}

        Answer:
        """
        response=llm.invoke(prompt)

        st.subheader("Answer:")
        st.write(response)



    
