import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph


load_dotenv()

# --- Load API keys: .env locally, st.secrets on Streamlit Cloud ---
def _get_secret(key: str) -> str:
    """Return an API key from env-vars first, then st.secrets."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""

# --- Constants ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
KNOWLEDGE_BASE_DIR = "knowledge-base"
FAISS_INDEX_DIR = "faiss_index"
LLM_MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ================================================================
# CACHED EMBEDDINGS — avoids re-downloading model on every rerun
# ================================================================
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ================================================================
# PREMIUM CSS STYLING — ALIGNMENT FIXED
# ================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* --- Global --- */
    html, body, .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, li, label, input { font-family: 'Inter', sans-serif; }
    * { box-sizing: border-box; }
    span[class*="material"], section[data-testid="stFileUploader"] button span > span:first-child > span { font-family: 'Material Symbols Rounded', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d0b1a 40%, #0a0a0f 100%);
    }

    /* --- Animated gradient background layer --- */
    .stApp::before {
        content: '';
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(124,58,237,0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(59,130,246,0.04) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(236,72,153,0.03) 0%, transparent 50%);
        pointer-events: none; z-index: 0;
        animation: bgShift 12s ease-in-out infinite alternate;
    }
    @keyframes bgShift {
        0% { opacity: 0.6; }
        100% { opacity: 1; }
    }

    /* ================================================================
       SIDEBAR — fixed alignment for all child elements
    ================================================================ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0b1a 0%, #111127 100%) !important;
        border-right: 1px solid rgba(124,58,237,0.15);
    }
    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem !important;
        display: flex;
        flex-direction: column;
        gap: 0;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        width: 100%;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c4b5fd !important;
        margin: 0 0 0.5rem 0;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        margin: 0 0 0.4rem 0;
        color: #9ca3af;
        font-size: 0.85rem;
    }

    /* ================================================================
       STATUS BADGES — full-width block alignment in sidebar
    ================================================================ */
    .badge-row {
        display: flex;
        align-items: center;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    .status-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 7px 14px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 500;
        width: 100%;
        line-height: 1;
    }
    .status-ready {
        background: rgba(34,197,94,0.08);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.2);
    }
    .status-waiting {
        background: rgba(250,204,21,0.08);
        color: #fbbf24;
        border: 1px solid rgba(250,204,21,0.2);
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .status-ready .status-dot {
        background: #4ade80;
        animation: pulse 2s infinite;
    }
    .status-waiting .status-dot { background: #fbbf24; }
    @keyframes pulse {
        0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74,222,128,0.5); }
        50% { opacity: 0.7; box-shadow: 0 0 0 5px rgba(74,222,128,0); }
    }

    /* ================================================================
       HERO SECTION — centered block with correct vertical rhythm
    ================================================================ */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 3rem 1rem 1.5rem;
        width: 100%;
        animation: fadeSlideUp 0.8s ease-out;
    }
    .hero-icon {
        font-size: 3.5rem;
        line-height: 1;
        margin-bottom: 0.75rem;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%,100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        line-height: 1.15;
        background: linear-gradient(135deg, #c4b5fd, #7c3aed, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 200% auto;
        animation: shimmer 3s linear infinite;
    }
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    .hero-sub {
        color: #9ca3af;
        font-size: 1rem;
        margin: 0;
        font-weight: 300;
        letter-spacing: 0.01em;
    }
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ================================================================
       FEATURE CARDS — proper grid with centered alignment
    ================================================================ */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.1rem;
        max-width: 860px;
        width: 100%;
        margin: 1.75rem auto 0;
        padding: 0;
    }
    @media (max-width: 720px) {
        .feature-grid { grid-template-columns: 1fr; }
    }
    .feature-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(124,58,237,0.12);
        border-radius: 14px;
        padding: 1.4rem 1.2rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 0.4rem;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        animation: fadeSlideUp 0.6s ease-out backwards;
    }
    .feature-card:nth-child(1) { animation-delay: 0.1s; }
    .feature-card:nth-child(2) { animation-delay: 0.22s; }
    .feature-card:nth-child(3) { animation-delay: 0.34s; }
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: rgba(124,58,237,0.32);
        box-shadow: 0 8px 28px rgba(124,58,237,0.1);
    }
    .card-icon  { font-size: 1.9rem; line-height: 1; }
    .card-title { font-weight: 600; color: #e8e8f0; font-size: 0.95rem; margin: 0; }
    .card-desc  { color: #6b7280; font-size: 0.8rem; margin: 0; line-height: 1.5; }

    /* ================================================================
       DIVIDER
    ================================================================ */
    .styled-divider {
        height: 1px;
        margin: 1rem 0;
        background: linear-gradient(90deg, transparent, rgba(124,58,237,0.3), transparent);
        border: none;
        width: 100%;
    }

    /* ================================================================
       SOURCE CARD — aligned flex row for sources
    ================================================================ */
    .source-card {
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 6px;
        background: rgba(124,58,237,0.06);
        border: 1px solid rgba(124,58,237,0.15);
        border-radius: 10px;
        padding: 0.65rem 1rem;
        margin-top: 0.75rem;
        font-size: 0.78rem;
        color: #9ca3af;
        line-height: 1.5;
    }
    .source-card strong {
        color: #c4b5fd;
        flex-shrink: 0;
        margin-right: 2px;
    }
    .source-card a {
        color: #7c3aed;
        word-break: break-all;
    }
    .source-sep {
        color: rgba(124,58,237,0.4);
        margin: 0 2px;
    }

    /* ================================================================
       CHAT MESSAGES — correct avatar + content alignment
    ================================================================ */
    .stChatMessage {
        display: flex !important;
        flex-direction: row !important;
        align-items: flex-start !important;
        gap: 0.75rem !important;
        border-radius: 14px !important;
        border: 1px solid rgba(124,58,237,0.08) !important;
        padding: 0.85rem 1rem !important;
        animation: fadeSlideUp 0.3s ease-out;
    }
    /* Avatar — fixed size, no shrink, top-aligned */
    .stChatMessage [data-testid="stChatMessageAvatarUser"],
    .stChatMessage [data-testid="stChatMessageAvatarAssistant"] {
        flex-shrink: 0 !important;
        width: 2rem !important;
        height: 2rem !important;
        border-radius: 8px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin-top: 0 !important;
        overflow: hidden !important;
    }
    /* Message content — takes remaining width, no overlap */
    .stChatMessage [data-testid="stChatMessageContent"] {
        flex: 1 !important;
        min-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    /* Markdown inside message — no extra top margin */
    .stChatMessage [data-testid="stChatMessageContent"] .stMarkdown {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .stChatMessage [data-testid="stChatMessageContent"] p:first-child {
        margin-top: 0 !important;
    }

    /* ================================================================
       BUTTONS — consistent sizing, no overflow
    ================================================================ */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        width: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 18px rgba(124,58,237,0.22) !important;
    }

    /* ================================================================
       FILE UPLOADER — clean single button, no double label
    ================================================================ */
    section[data-testid="stFileUploader"] {
        width: 100% !important;
    }
    /* The drop-zone wrapper */
    section[data-testid="stFileUploader"] > div {
        border: 2px dashed rgba(124,58,237,0.25) !important;
        border-radius: 10px !important;
        background: rgba(255,255,255,0.02) !important;
        transition: border-color 0.3s, background 0.3s;
        padding: 1rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    section[data-testid="stFileUploader"] > div:hover {
        border-color: rgba(124,58,237,0.5) !important;
        background: rgba(124,58,237,0.04) !important;
    }
    /* Hide the redundant text label that causes the double-text */
    section[data-testid="stFileUploader"] span[data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child {
        display: none !important;
    }
    /* Style the Browse button inside uploader */
    section[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] {
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        padding: 6px 16px !important;
        border: 1px solid rgba(124,58,237,0.4) !important;
        background: rgba(124,58,237,0.1) !important;
        color: #c4b5fd !important;
        width: auto !important;
    }
    
    /* ================================================================
       INDEXED FILES LIST
    ================================================================ */
    .file-list {
        list-style: none;
        padding: 0;
        margin: 0.4rem 0 0;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .file-list li {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        color: #9ca3af;
        background: rgba(255,255,255,0.03);
        border-radius: 6px;
        padding: 5px 8px;
        border: 1px solid rgba(124,58,237,0.08);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """, unsafe_allow_html=True)


# ================================================================
# PDF INGESTION
# ================================================================
def ingest_pdfs_into_vectordb():
    documents = []
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        return 0
    for file_name in os.listdir(KNOWLEDGE_BASE_DIR):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(KNOWLEDGE_BASE_DIR, file_name)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    if not documents:
        return 0
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    embeddings = get_embeddings()
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR, ignore_errors=True)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)
    return len(texts)


# ================================================================
# RETRIEVER
# ================================================================
def create_retriever():
    if not os.path.exists(FAISS_INDEX_DIR):
        return None
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


# ================================================================
# LANGGRAPH STATE
# ================================================================
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    sender: str
    answer: str
    chat_history: str


# ================================================================
# GRAPH NODES
# ================================================================
def router_node(state: GraphState) -> str:
    question = state["question"]
    has_vectorstore = os.path.exists(FAISS_INDEX_DIR)
    if not has_vectorstore:
        return "web_search"
    prompt = f"""You are a routing assistant. Decide the best data source for the question.

Available sources:
- "vectorstore": Search uploaded PDF documents for the answer.
- "web_search": Search the internet for current/live information.

Rules:
- If the question is about general knowledge, current events, news, or live data -> "web_search"
- If the question seems related to document content, uploaded files, or specific technical material -> "vectorstore"
- When unsure, prefer "vectorstore" if documents are available.

Question: {question}

Respond with ONLY one word: either "vectorstore" or "web_search"
"""
    llm = ChatGroq(temperature=0, model_name=LLM_MODEL_ID)
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    return "web_search" if "web_search" in decision else "vectorstore"


def retrieve_node(state: GraphState) -> GraphState:
    retriever = create_retriever()
    if retriever is None:
        return {"documents": [], "sender": "retrieve"}
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "sender": "retrieve"}


def web_search_node(state: GraphState) -> GraphState:
    try:
        tavily = TavilySearch(max_results=3, search_depth="advanced")
        results = tavily.invoke(state["question"])
        docs = []
        if isinstance(results, list):
            for r in results:
                docs.append(Document(page_content=r.get("content", ""), metadata={"source": r.get("url", "web")}))
        elif isinstance(results, dict) and "results" in results:
            for r in results["results"]:
                docs.append(Document(page_content=r.get("content", ""), metadata={"source": r.get("url", "web")}))
        elif isinstance(results, str):
            docs.append(Document(page_content=results, metadata={"source": "web_search"}))
        return {"documents": docs, "sender": "web_search"}
    except Exception as e:
        return {"documents": [Document(page_content=f"Web search failed: {str(e)}", metadata={"source": "error"})], "sender": "web_search"}


def generate_node(state: GraphState) -> GraphState:
    context = "\n\n".join(doc.page_content for doc in state.get("documents", []))
    chat_history = state.get("chat_history", "")
    prompt = f"""You are a knowledgeable AI assistant. Answer the user's question using the provided context.

Rules:
- Use the context to give accurate, detailed answers.
- If the context doesn't contain the answer, say so honestly and provide your best general knowledge.
- Be concise but thorough. Use markdown formatting for readability.
- Reference the source when possible (PDF page or web URL).

Previous conversation:
{chat_history}

Retrieved context:
{context}

Current question: {state["question"]}

Answer:"""
    llm = ChatGroq(temperature=0.1, model_name=LLM_MODEL_ID)
    response = llm.invoke(prompt)
    return {"answer": response.content, "documents": state.get("documents", []), "sender": "generate"}


# ================================================================
# GRAPH BUILDER
# ================================================================

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)
    graph.add_conditional_edges(START, router_node, {"vectorstore": "retrieve", "web_search": "web_search"})
    graph.add_edge("retrieve", "generate")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# ================================================================
# CHAT HISTORY
# ================================================================
def build_chat_history() -> str:
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        return ""
    recent = st.session_state.messages[-6:]
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


# ================================================================
# MAIN UI
# ================================================================
def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="RAG Agent | PDF + Web Search",
        page_icon="🧠",
        layout="wide"
    )
    inject_css()

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### 🧠 RAG Agent")
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        groq_key   = _get_secret("GROQ_API_KEY")
        tavily_key = _get_secret("TAVILY_API_KEY")

        # Ensure env vars are set for libraries that read them directly
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        # --- PDF Upload ---
        st.markdown("**📄 Upload PDFs**")
        uploaded_files = st.file_uploader(
            "Browse files",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
            # label_visibility="show"
        )

        if uploaded_files:
            if st.button("⚡ Process & Index", use_container_width=True, type="primary"):
                os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
                for file in uploaded_files:
                    path = os.path.join(KNOWLEDGE_BASE_DIR, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                with st.spinner("Chunking & embedding PDFs..."):
                    count = ingest_pdfs_into_vectordb()
                if count > 0:
                    st.success(f"✅ {count} chunks indexed!")
                    st.session_state["pdf_ready"] = True
                else:
                    st.error("No text extracted from PDFs.")

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # --- Vector DB Status ---
        db_exists = os.path.exists(FAISS_INDEX_DIR)
        st.markdown(
            f'<div class="badge-row">'
            f'<div class="status-badge {"status-ready" if db_exists else "status-waiting"}">'
            f'<span class="status-dot"></span>'
            f'{"Vector DB Ready" if db_exists else "No PDFs indexed yet"}'
            f'</div></div>',
            unsafe_allow_html=True
        )

        # List indexed files
        if db_exists and os.path.exists(KNOWLEDGE_BASE_DIR):
            files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.lower().endswith(".pdf")]
            if files:
                st.markdown(f"**{len(files)} PDF(s) indexed:**")
                items_html = "".join(f'<li>📄 {f}</li>' for f in files)
                st.markdown(f'<ul class="file-list">{items_html}</ul>', unsafe_allow_html=True)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # --- Reset Buttons ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                st.session_state.messages = []
                if os.path.exists(FAISS_INDEX_DIR):
                    shutil.rmtree(FAISS_INDEX_DIR, ignore_errors=True)
                if os.path.exists(KNOWLEDGE_BASE_DIR):
                    shutil.rmtree(KNOWLEDGE_BASE_DIR, ignore_errors=True)
                st.session_state["pdf_ready"] = False
                st.rerun()

    # ---- Main Content ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hero (empty chat only)
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="hero-container">
            <div class="hero-icon">🧠</div>
            <h1 class="hero-title">RAG Agent</h1>
            <p class="hero-sub">Upload PDFs and search the web — powered by Groq LLM</p>
        </div>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="card-icon">📄</div>
                <div class="card-title">PDF Intelligence</div>
                <div class="card-desc">Upload documents and ask questions. AI retrieves precise answers from your files.</div>
            </div>
            <div class="feature-card">
                <div class="card-icon">🌐</div>
                <div class="card-title">Web Search</div>
                <div class="card-desc">Automatically searches the internet when your question needs live information.</div>
            </div>
            <div class="feature-card">
                <div class="card-icon">🔀</div>
                <div class="card-title">Smart Routing</div>
                <div class="card-desc">AI decides the best source — PDFs or web — for every question you ask.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Chat Messages ----
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("sources"):
                items = []
                for s in m["sources"]:
                    if s.startswith("http"):
                        items.append(f'<a href="{s}" target="_blank">{s[:55]}…</a>')
                    else:
                        items.append(s)
                sep = '<span class="source-sep">·</span>'
                st.markdown(
                    f'<div class="source-card"><strong>Sources:</strong> {sep.join(items)}</div>',
                    unsafe_allow_html=True
                )

    # ---- Chat Input ----
    if question := st.chat_input("Ask anything — PDFs or the web..."):
        if not groq_key or groq_key.startswith("gsk_your"):
            st.error("Please set your `GROQ_API_KEY` in the `.env` file.")
            return

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    app = build_graph()
                    result = None
                    for output in app.stream(
                        {"question": question, "chat_history": build_chat_history()},
                        stream_mode="values"
                    ):
                        result = output

                    answer    = result.get("answer", "I couldn't generate an answer. Please try again.")
                    documents = result.get("documents", [])

                    st.markdown(answer)

                    sources = []
                    for doc in documents:
                        src = doc.metadata.get("source", "")
                        if src and src not in sources and src != "error":
                            sources.append(src)

                    if sources:
                        items = []
                        for s in sources[:5]:
                            if s.startswith("http"):
                                items.append(f'<a href="{s}" target="_blank">{s[:55]}…</a>')
                            else:
                                items.append(s)
                        sep = '<span class="source-sep">·</span>'
                        st.markdown(
                            f'<div class="source-card"><strong>Sources:</strong> {sep.join(items)}</div>',
                            unsafe_allow_html=True
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


main()