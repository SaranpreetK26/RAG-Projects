import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

import tempfile
import re

# --------------------------
# PAGE CONFIG
# --------------------------

st.set_page_config(
    page_title="College Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 College Knowledge Assistant")

# --------------------------
# SESSION STATE
# --------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None


# --------------------------
# CLEANER
# --------------------------

def clean_text(text):
    text = re.sub(r"<pad>", "", text)
    text = re.sub(r"<EOS>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------
# EMBEDDINGS
# --------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------
# LLM
# --------------------------

@st.cache_resource
def load_llm():
    return ChatOllama(
        model="gemma2:2b",
        temperature=0
    )


# --------------------------
# BUILD VECTOR DB
# --------------------------

def build_database(files):

    docs = []

    for file in files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:

            temp.write(file.read())

        loader = PyPDFLoader(temp.name)

        pdf_docs = loader.load()

        for doc in pdf_docs:
            doc.page_content = clean_text(
                doc.page_content
            )

        docs.extend(pdf_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = load_embeddings()

    vector_db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_db"
    )

    return vector_db


# --------------------------
# CHAT FUNCTION
# --------------------------

def ask_question(question):

    docs = st.session_state.db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a college assistant.

Answer ONLY from the provided context.

If the answer is unavailable say:

I could not find that information in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""

    llm = load_llm()

    response = llm.invoke(prompt)

    sources = []

    for doc in docs:

        src = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            0
        )

        sources.append(
            f"{src} (Page {page+1})"
        )

    return response.content, list(set(sources))


# --------------------------
# SIDEBAR
# --------------------------

with st.sidebar:

    st.header("📚 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Build Knowledge Base"):

        if uploaded_files:

            with st.spinner(
                "Processing..."
            ):

                st.session_state.db = build_database(
                    uploaded_files
                )

            st.success(
                "Knowledge Base Ready!"
            )


# --------------------------
# DISPLAY CHAT HISTORY
# --------------------------

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )


# --------------------------
# CHAT INPUT
# --------------------------

if prompt := st.chat_input(
    "Ask a question..."
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.db is None:

        answer = (
            "Please upload PDFs and build the knowledge base first."
        )

        with st.chat_message(
            "assistant"
        ):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    else:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                answer, sources = ask_question(
                    prompt
                )

                st.markdown(answer)

                st.markdown("### Sources")

                for source in sources:
                    st.write(f"📄 {source}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content":
                f"{answer}\n\nSources:\n"
                + "\n".join(sources)
            }
        )ś