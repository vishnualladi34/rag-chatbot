import os
import re
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# ------------------------------------------------------------------
# Sidebar: API key, PDF upload, settings
# ------------------------------------------------------------------
def get_api_key():
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = None
    return key or os.environ.get("GROQ_API_KEY") or st.sidebar.text_input(
        "Groq API key", type="password"
    )


api_key = get_api_key()
uploaded = st.sidebar.file_uploader("Upload a PDF", type="pdf")
top_k = st.sidebar.slider("Chunks to retrieve (k)", 2, 8, 4)

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

if not api_key or not uploaded:
    st.info("Add your Groq API key and upload a PDF in the sidebar to start.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------
# For clause-numbered documents (e.g. HR-1.2, FIN-3.4) we keep each topic's
# clauses together in one chunk, with the topic name on top and the page
# footer removed. Other PDFs fall back to a normal recursive splitter.
HEADER = re.compile(r"Nexora Employee Policy Handbook 2026\s*\|\s*Page \d+")
CLAUSE = re.compile(
    r"(?m)^(([A-Z]{2,4}-\d+)\.\d+)\s+(.*?)"
    r"(?=^[A-Z]{2,4}-\d+\.\d+\s|^[A-Z]{2,4}-\d+\.\s|\Z)",
    re.S,
)
TOPIC = re.compile(r"(?m)^([A-Z]{2,4}-\d+)\.\s+(.+)$")


def clause_chunks(pages):
    topic_names, topic_lines, topic_page = {}, {}, {}
    for p in pages:
        text = HEADER.sub("", p.page_content)
        for m in TOPIC.finditer(text):
            topic_names[m.group(1)] = m.group(2).strip()
        for m in CLAUSE.finditer(text):
            clause_id, topic_id = m.group(1), m.group(2)
            body = " ".join(m.group(3).split())
            topic_lines.setdefault(topic_id, []).append(f"{clause_id} {body}")
            topic_page.setdefault(topic_id, p.metadata.get("page", 0))
    return [
        Document(
            page_content=f"[{tid}] {topic_names.get(tid, '')}\n" + "\n".join(lines),
            metadata={"page": topic_page[tid], "topic": tid},
        )
        for tid, lines in topic_lines.items()
    ]


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Reading and indexing the PDF...")
def build_vectorstore(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        path = f.name
    pages = PyPDFLoader(path).load()

    chunks = clause_chunks(pages)
    mode = "clause-aware"
    if len(chunks) < 5:  # not a clause-numbered document
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=150
        ).split_documents(pages)
        mode = "recursive 800/150"

    vs = FAISS.from_documents(chunks, get_embeddings())
    return vs, mode, len(chunks), len(pages)


vectorstore, mode, n_chunks, n_pages = build_vectorstore(uploaded.getvalue())
st.sidebar.caption(f"{n_pages} pages -> {n_chunks} chunks ({mode})")

retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": top_k, "fetch_k": max(20, top_k * 4)}
)

# ------------------------------------------------------------------
# LLM, prompts, chains
# ------------------------------------------------------------------
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering questions about a document.\n"
     "Use ONLY the context below. If the answer is not in the context, say: "
     "\"I couldn't find that in the document.\" Do not make things up.\n"
     "If several topics could match the question, say which topic you are answering about.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest question, rewrite the question so it "
     "makes sense on its own. Do NOT answer it. If it is already clear, return it unchanged."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

answer_chain = qa_prompt | llm | StrOutputParser()
condense_chain = condense_prompt | llm | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}] {d.page_content}" for d in docs
    )


def to_lc_history(messages):
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in messages
    ]


# ------------------------------------------------------------------
# Chat UI
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.caption(s)

question = st.chat_input("Ask something about the PDF")

if question:
    history = to_lc_history(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                standalone = (
                    condense_chain.invoke({"history": history, "question": question})
                    if history else question
                )
                docs = retriever.invoke(standalone)
                answer = answer_chain.invoke({
                    "context": format_docs(docs),
                    "history": history,
                    "question": question,
                })
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        st.write(answer)

        sources = [
            f"Page {d.metadata.get('page', 0) + 1}: "
            f"{d.page_content[:200].replace(chr(10), ' ')}..."
            for d in docs
        ]
        with st.expander("Sources"):
            for s in sources:
                st.caption(s)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
