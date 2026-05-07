import os
import sys
import tempfile
import streamlit as st

# Fix imports when running: streamlit run vector_store/app.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.prompts import ChatPromptTemplate
from vector_store.document_selector import get_available_documents
from retrievers.hybrid_retriever import retrieve_with_multiquery
from retrievers.citation_builder import format_citations_block
from document_loaders.document_registry import register_document


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind · RAG Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

html, body, [class*="css"] {
    background-color: #0e0f14 !important;
    color: #d4c9b8 !important;
}

.main .block-container {
    padding: 2rem 2.5rem 4rem;
    max-width: 950px;
}

[data-testid="stSidebar"] {
    background: #13141a !important;
    border-right: 1px solid #2a2b35 !important;
}

[data-testid="stSidebar"] * {
    color: #b0a898 !important;
}

h1 {
    font-family: 'Syne', sans-serif !important;
    color: #f0e6d3 !important;
}

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3a3b4a;
    margin: 20px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2030;
}

.user-bubble {
    background: #1c1e2a;
    border: 1px solid #2e3045;
    border-radius: 12px 12px 2px 12px;
    padding: 14px 18px;
    margin: 10px 0 10px 60px;
    font-family: 'Lora', serif;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #d4c9b8;
}

.ai-bubble {
    background: #161820;
    border: 1px solid #2a2c3e;
    border-left: 3px solid #c8a96e;
    border-radius: 2px 12px 12px 12px;
    padding: 16px 20px;
    margin: 10px 60px 10px 0;
    font-family: 'Lora', serif;
    font-size: 0.95rem;
    line-height: 1.8;
    color: #d4c9b8;
}

.ai-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    color: #c8a96e;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.user-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    color: #6e8ec8;
    text-transform: uppercase;
    margin-bottom: 6px;
    text-align: right;
}

.chunk-card {
    background: #12131a;
    border: 1px solid #1e2030;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #8a8f9f;
    line-height: 1.6;
}

.stat-box {
    background: #12131a;
    border: 1px solid #1e2030;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}

.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #c8a96e;
}

.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    color: #3a3b4a;
    text-transform: uppercase;
}

.stButton > button {
    background: linear-gradient(135deg, #c8a96e, #a07840) !important;
    color: #0e0f14 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Lazy imports ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_imports():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    return PyPDFLoader, RecursiveCharacterTextSplitter, Chroma


@st.cache_resource
def get_embedder():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_llm():
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline

    text_pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
    )

    return HuggingFacePipeline(pipeline=text_pipe)


# ── Session state ────────────────────────────────────────────────────────────
for key, value in {
    "messages": [],
    "vectorstore": None,
    "db_ready": False,
    "doc_stats": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ── Sidebar ──────────────────────────────────────────────────────────────────
documents = get_available_documents()

with st.sidebar:
    st.markdown("""
    <div style='padding:18px 0 10px'>
      <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                  color:#f0e6d3;letter-spacing:-0.5px;'>DocMind</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                  letter-spacing:2px;color:#3a3b4a;text-transform:uppercase;
                  margin-top:2px;'>RAG · PDF Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-label">Knowledge Base</div>', unsafe_allow_html=True)

    selected_document = st.selectbox(
        "📚 Select Document",
        documents,
        index=0
    )

    st.caption(f"Indexed documents: {max(len(documents) - 1, 0)}")

    st.divider()

    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)

    if st.session_state.db_ready or len(documents) > 1:
        st.success("DB Ready")
    else:
        st.error("No Database")

    st.divider()

    st.markdown('<div class="section-label">Retrieval Settings</div>', unsafe_allow_html=True)

    chunk_size = st.slider("Chunk size", 300, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
    k_docs = st.slider("Docs to retrieve (k)", 1, 8, 4)
    fetch_k = st.slider("Fetch k (MMR pool)", k_docs, 20, 10)

    st.divider()

    st.markdown('<div class="section-label">Model Stack</div>', unsafe_allow_html=True)

    st.markdown("""
    **Embedding:** all-MiniLM-L6-v2  
    **Vector DB:** ChromaDB  
    **Retrieval:** Multi-query + MMR  
    **LLM:** TinyLlama-1.1B-Chat  
    """)

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("⚡ Reset Session"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.db_ready = False
        st.session_state.doc_stats = {}
        st.rerun()


# ── Main area ────────────────────────────────────────────────────────────────
st.markdown("# 📖 DocMind RAG Book Assistant")
st.markdown("""
Upload PDFs, index them into ChromaDB, select a document, and ask grounded questions.
""")


# ── Upload and index PDF ─────────────────────────────────────────────────────
with st.expander("📂 Upload & Index PDF", expanded=not st.session_state.db_ready):
    uploaded = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="stat-box">
              <div class="stat-num">{round(uploaded.size / 1024, 1)}</div>
              <div class="stat-label">KB size</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-box">
              <div class="stat-num">{chunk_size}</div>
              <div class="stat-label">chunk size</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stat-box">
              <div class="stat-num">{chunk_overlap}</div>
              <div class="stat-label">overlap</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⚙ Build Knowledge Base"):
            try:
                PyPDFLoader, RecursiveCharacterTextSplitter, Chroma = get_imports()

                with st.spinner("Loading PDF…"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(uploaded.read())
                        tmp_path = f.name

                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                with st.spinner("Splitting into chunks…"):
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    chunks = splitter.split_documents(docs)

                    # Phase 3 metadata: required for document filtering and citations
                    for i, chunk in enumerate(chunks):
                        chunk.metadata["source"] = uploaded.name
                        chunk.metadata["chunk_id"] = i + 1

                with st.spinner("Embedding and storing in ChromaDB…"):
                    embeddings = get_embedder()

                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory="chroma_db",
                    )

                # Phase 3 registry: required for document selector
                register_document(
                    file_path=uploaded.name,
                    pages=len(docs),
                    chunks=len(chunks),
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.db_ready = True
                st.session_state.doc_stats = {
                    "pages": len(docs),
                    "chunks": len(chunks),
                    "filename": uploaded.name,
                }
                st.session_state.messages = []

                os.unlink(tmp_path)

                st.success(f"✅ Indexed **{len(docs)} pages** → **{len(chunks)} chunks** into ChromaDB")
                st.info("Refresh the app once if the new document does not appear in the selector.")

                with st.expander("Preview first 3 chunks"):
                    for i, chunk in enumerate(chunks[:3]):
                        st.markdown(
                            f"""
                            <div class="chunk-card">
                                <b>Chunk {i + 1}</b> · Source: {chunk.metadata.get("source", "?")}<br>
                                {chunk.page_content[:300]}…
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.error(f"Error building database: {e}")


# ── Chat interface ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-label" style="margin-top:32px;">Conversation</div>',
    unsafe_allow_html=True,
)

has_docs = st.session_state.db_ready or len(documents) > 1

if not has_docs:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;color:#555;font-size:1rem;'>
      ↑ Upload and index a PDF to begin chatting
    </div>
    """, unsafe_allow_html=True)

else:
    current_doc = selected_document if selected_document else "All Documents"

    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                color:#3a4a5a;margin-bottom:20px;'>
      Chatting with · <span style='color:#c8a96e;'>{current_doc}</span>
      &nbsp;·&nbsp; Multi-query + MMR
    </div>
    """, unsafe_allow_html=True)

    # Display old messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-label">You</div>
            <div class="user-bubble">{msg["content"]}</div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="ai-label">DocMind</div>
            <div class="ai-bubble">{msg["content"]}</div>
            """, unsafe_allow_html=True)

            if msg.get("sources"):
                with st.expander(f"📎 Sources ({len(msg['sources'])} chunks)"):
                    for i, source in enumerate(msg["sources"]):
                        st.markdown(
                            f"""
                            <div class="chunk-card">
                                <b>Chunk {i + 1}</b> ·
                                Source: {source.metadata.get("source", "?")} ·
                                Page: {source.metadata.get("page", "?")}<br>
                                {source.page_content[:350]}…
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])

        with col_in:
            user_input = st.text_input(
                "Ask a question",
                placeholder="Ask something from your selected document...",
                label_visibility="collapsed",
            )

        with col_btn:
            submitted = st.form_submit_button("Ask →")

    # Submit logic
    if submitted:
        if user_input.strip():
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
            })

            try:
                with st.spinner("Retrieving context & generating answer…"):
                    retrieved_docs = retrieve_with_multiquery(
                        user_input,
                        selected_document,
                    )

                    context = "\n\n".join([d.page_content for d in retrieved_docs])
                    citations = format_citations_block(retrieved_docs)

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are a helpful document QA assistant.
Use only the provided context to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the document."
Be clear, concise, and include source references when available."""),
                        ("human", """Context:
{context}

Question:
{question}

Sources:
{citations}
"""),
                    ])

                    final_prompt = prompt.format(
                        context=context,
                        question=user_input,
                        citations=citations,
                    )

                    llm = get_llm()
                    answer = llm.invoke(final_prompt)
                    answer_text = answer.content if hasattr(answer, "content") else str(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": retrieved_docs,
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {e}",
                    "sources": [],
                })

            st.rerun()

        else:
            st.warning(" Please enter a question before submitting.")