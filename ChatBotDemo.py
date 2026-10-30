import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

@st .cache_data
def load_vectorstore():
    loader = TextLoader("CPP.txt", encoding="utf8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    final_document=text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(final_document, embeddings)
    return db

db = load_vectorstore()

query=st.text_input("Ask a question about C++:")

if query:
    docs=db.similarity_search(query, k=3)
    st.subheader("Retrived Context:")
    for i, doc in enumerate(docs):
        st.markdown(f"**Result {i+1}:**")
        st.write(doc.page_content)