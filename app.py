from pathlib import Path

import streamlit as st

from src.config import get_settings
from src.ingestion import build_index
from src.rag import InsuranceRAG

st.set_page_config(
    page_title="Document Intelligence Copilot",
    page_icon="📚",
    layout="wide",
)

settings = get_settings()
DATA_DIR = Path(settings.data_dir)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_agent():
    return InsuranceRAG(settings)


agent = load_agent()

st.title("📚 Document Intelligence Copilot")
st.caption("A source-aware RAG assistant for asking questions about your own PDF documents.")

with st.sidebar:
    st.subheader("Knowledge base")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Your PDFs stay in the app's data directory and are used as the retrieval knowledge base.",
    )

    if uploaded_files and st.button("Build / refresh index", type="primary"):
        for uploaded_file in uploaded_files:
            (DATA_DIR / uploaded_file.name).write_bytes(uploaded_file.getbuffer())

        with st.spinner("Building the document index..."):
            try:
                chunks = build_index()
                agent = load_agent()
                st.success(f"Indexed {chunks} text chunks.")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    st.subheader("Try a question")
    examples = [
        "What is risk management?",
        "What is indemnity?",
        "Summarize the main eligibility rules in these documents.",
        "What are the key exclusions mentioned in the documents?",
    ]
    selected = st.radio("Examples", ["None"] + examples, index=0)

    st.divider()
    st.write(f"**Indexed chunks:** {agent.document_count}")
    st.write("**Retriever:** Chroma")
    st.write(f"**Embedding:** {settings.embedding_model.split('/')[-1]}")
    st.write(f"**Top-k:** {settings.top_k}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(source)

question = selected if selected != "None" else None
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    question = prompt

if question:
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        (m["role"], m["content"])
        for m in st.session_state.messages[-settings.max_history_turns * 2 :]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            result = agent.answer(question, history=history)

        st.markdown(result.answer)

        if result.sources:
            with st.expander("Sources"):
                for source in result.sources:
                    st.write(source)

        if result.confidence_label:
            st.caption(f"Retrieval signal: {result.confidence_label}")

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": result.sources,
        }
    )
