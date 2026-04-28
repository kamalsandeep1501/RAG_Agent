import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph

# --- Page Config ---
st.set_page_config(
    page_title="RAG Agent | PDF + Web Search",
    page_icon="🧠",
    layout="wide"
)

load_dotenv()

# --- Constants ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
KNOWLEDGE_BASE_DIR = "knowledge-base"
PERSIST_DIRECTORY = "chroma_db"
LLM_MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ================================================================
# PREMIUM CSS STYLING
# ================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* --- Global --- */
    * { font-family: 'Inter', sans-serif !important; }
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

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0b1a 0%, #111127 100%) !important;
        border-right: 1px solid rgba(124,58,237,0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c4b5fd !important;
    }

    /* --- Hero Section --- */
    .hero-container {
        text-align: center; padding: 3rem 1rem 2rem;
        animation: fadeSlideUp 0.8s ease-out;
    }
    .hero-icon {
        font-size: 4rem; margin-bottom: 0.5rem;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%,100% { transform: translateY(0); }
        50% { transform: translateY(-12px); }
    }
    .hero-title {
        font-size: 2.8rem; font-weight: 800; margin: 0;
        background: linear-gradient(135deg, #c4b5fd, #7c3aed, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-size: 200% auto;
        animation: shimmer 3s linear infinite;
    }
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    .hero-sub {
        color: #9ca3af; font-size: 1.1rem; margin-top: 0.5rem; font-weight: 300;
    }
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- Feature Cards --- */
    .feature-grid {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem;
        max-width: 900px; margin: 2rem auto; padding: 0 1rem;
    }
    .feature-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(124,58,237,0.12);
        border-radius: 16px; padding: 1.5rem; text-align: center;
        transition: all 0.3s ease;
        animation: fadeSlideUp 0.6s ease-out backwards;
    }
    .feature-card:nth-child(1) { animation-delay: 0.1s; }
    .feature-card:nth-child(2) { animation-delay: 0.25s; }
    .feature-card:nth-child(3) { animation-delay: 0.4s; }
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: rgba(124,58,237,0.35);
        box-shadow: 0 8px 30px rgba(124,58,237,0.1);
    }
    .feature-card .card-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .feature-card .card-title { font-weight: 600; color: #e8e8f0; font-size: 1rem; }
    .feature-card .card-desc { color: #6b7280; font-size: 0.82rem; margin-top: 0.3rem; line-height: 1.4; }

    /* --- Status badges --- */
    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 500;
    }
    .status-ready {
        background: rgba(34,197,94,0.1); color: #4ade80;
        border: 1px solid rgba(34,197,94,0.2);
    }
    .status-waiting {
        background: rgba(250,204,21,0.1); color: #fbbf24;
        border: 1px solid rgba(250,204,21,0.2);
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%; display: inline-block;
    }
    .status-ready .status-dot {
        background: #4ade80;
        animation: pulse 2s infinite;
    }
    .status-waiting .status-dot { background: #fbbf24; }
    @keyframes pulse {
        0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74,222,128,0.5); }
        50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(74,222,128,0); }
    }

    /* --- Source card in chat --- */
    .source-card {
        background: rgba(124,58,237,0.06);
        border: 1px solid rgba(124,58,237,0.15);
        border-radius: 10px; padding: 0.7rem 1rem; margin-top: 0.8rem;
        font-size: 0.8rem; color: #9ca3af;
    }
    .source-card strong { color: #c4b5fd; }

    /* --- Chat styling --- */
    .stChatMessage {
        border-radius: 16px !important;
        border: 1px solid rgba(124,58,237,0.08) !important;
        animation: fadeSlideUp 0.35s ease-out;
    }

    /* --- Buttons --- */
    .stButton > button {
        border-radius: 12px !important; font-weight: 600 !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124,58,237,0.25) !important;
    }

    /* --- Divider --- */
    .styled-divider {
        height: 1px; margin: 1.5rem 0;
        background: linear-gradient(90deg, transparent, rgba(124,58,237,0.3), transparent);
    }

    /* --- Upload area --- */
    section[data-testid="stFileUploader"] {
        border: 2px dashed rgba(124,58,237,0.25) !important;
        border-radius: 12px !important;
        transition: border-color 0.3s;
    }
    section[data-testid="stFileUploader"]:hover {
        border-color: rgba(124,58,237,0.5) !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ================================================================
# PDF INGESTION — loads PDFs, chunks them, stores in ChromaDB
# ================================================================
def ingest_pdfs_into_vectordb():
    """Load all PDFs from KNOWLEDGE_BASE_DIR, split into chunks, embed, and store in ChromaDB."""
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Remove old DB if exists to avoid stale data
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    vectorstore = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return len(texts)


# ================================================================
# RETRIEVER — connects to existing ChromaDB for similarity search
# ================================================================
def create_retriever():
    """Create a retriever from the persisted ChromaDB vector store."""
    if not os.path.exists(PERSIST_DIRECTORY):
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


# ================================================================
# LANGGRAPH STATE
# ================================================================
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    sender: str
    answer: str
    chat_history: str  # conversational context


# ================================================================
# GRAPH NODES
# ================================================================

def router_node(state: GraphState) -> str:
    """LLM-based router: decides whether to search PDFs or the web."""
    question = state["question"]
    has_vectorstore = os.path.exists(PERSIST_DIRECTORY)

    # If no PDFs uploaded, always search the web
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
    """Retrieve relevant chunks from the ChromaDB vector store."""
    retriever = create_retriever()

    if retriever is None:
        return {"documents": [], "sender": "retrieve"}

    docs = retriever.invoke(state["question"])
    return {"documents": docs, "sender": "retrieve"}


def web_search_node(state: GraphState) -> GraphState:
    """Search the web using Tavily and return results as Documents."""
    try:
        tavily = TavilySearch(
            max_results=3,
            search_depth="advanced",
        )
        results = tavily.invoke(state["question"])

        docs = []
        # TavilySearch returns a list of dicts or a dict with "results"
        if isinstance(results, list):
            for r in results:
                content = r.get("content", "")
                url = r.get("url", "web")
                docs.append(Document(page_content=content, metadata={"source": url}))
        elif isinstance(results, dict) and "results" in results:
            for r in results["results"]:
                content = r.get("content", "")
                url = r.get("url", "web")
                docs.append(Document(page_content=content, metadata={"source": url}))
        elif isinstance(results, str):
            docs.append(Document(page_content=results, metadata={"source": "web_search"}))

        return {"documents": docs, "sender": "web_search"}

    except Exception as e:
        return {
            "documents": [Document(page_content=f"Web search failed: {str(e)}", metadata={"source": "error"})],
            "sender": "web_search"
        }


def generate_node(state: GraphState) -> GraphState:
    """Generate an answer using retrieved context and conversation history."""
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

    return {
        "answer": response.content,
        "documents": state.get("documents", []),
        "sender": "generate"
    }


# ================================================================
# GRAPH BUILDER
# ================================================================
def build_graph():
    """Construct the LangGraph RAG pipeline: Router -> Retrieve/WebSearch -> Generate."""
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)

    graph.add_conditional_edges(
        START,
        router_node,
        {
            "vectorstore": "retrieve",
            "web_search": "web_search"
        }
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ================================================================
# BUILD CHAT HISTORY STRING from session messages
# ================================================================
def build_chat_history() -> str:
    """Convert session messages into a formatted string for context."""
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        return ""

    # Keep last 6 messages for context window management
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
    inject_css()

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### 🧠 RAG Agent")
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # --- API Key Status ---
        groq_key = os.getenv("GROQ_API_KEY", "")
        tavily_key = os.getenv("TAVILY_API_KEY", "")

        st.markdown("**API Status**")
        if groq_key and not groq_key.startswith("gsk_your"):
            st.markdown('<span class="status-badge status-ready"><span class="status-dot"></span>Groq Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-waiting"><span class="status-dot"></span>Groq Key Missing</span>', unsafe_allow_html=True)

        if tavily_key and not tavily_key.startswith("tvly-your"):
            st.markdown('<span class="status-badge status-ready"><span class="status-dot"></span>Tavily Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-waiting"><span class="status-dot"></span>Tavily Key Missing</span>', unsafe_allow_html=True)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # --- PDF Upload ---
        st.markdown("**📄 Upload PDFs**")
        uploaded_files = st.file_uploader(
            "Drop your PDF files here",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if uploaded_files:
            if st.button("⚡ Process & Index", use_container_width=True, type="primary"):
                os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

                # Save uploaded files to disk
                for file in uploaded_files:
                    path = os.path.join(KNOWLEDGE_BASE_DIR, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())

                # Ingest into vector store
                with st.spinner("Chunking & embedding PDFs..."):
                    count = ingest_pdfs_into_vectordb()

                if count > 0:
                    st.success(f"{count} chunks indexed!")
                    st.session_state["pdf_ready"] = True
                else:
                    st.error("No text extracted from PDFs.")

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # --- Vector DB Status ---
        if os.path.exists(PERSIST_DIRECTORY):
            st.markdown('<span class="status-badge status-ready"><span class="status-dot"></span>Vector DB Ready</span>', unsafe_allow_html=True)

            # List indexed files
            if os.path.exists(KNOWLEDGE_BASE_DIR):
                files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.lower().endswith(".pdf")]
                if files:
                    st.markdown(f"**{len(files)} PDF(s) indexed:**")
                    for f_name in files:
                        st.markdown(f"- 📄 {f_name}")
        else:
            st.markdown('<span class="status-badge status-waiting"><span class="status-dot"></span>No PDFs indexed yet</span>', unsafe_allow_html=True)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # --- Reset ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                st.session_state.messages = []
                if os.path.exists(PERSIST_DIRECTORY):
                    shutil.rmtree(PERSIST_DIRECTORY)
                if os.path.exists(KNOWLEDGE_BASE_DIR):
                    shutil.rmtree(KNOWLEDGE_BASE_DIR)
                st.session_state["pdf_ready"] = False
                st.rerun()

    # ---- Main Content ----
    # Hero section (only when chat is empty)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="hero-container">
            <div class="hero-icon">🧠</div>
            <h1 class="hero-title">RAG Agent</h1>
            <p class="hero-sub">Upload PDFs and search the web — powered by Groq LLM</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
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
            # Show sources if available
            if m["role"] == "assistant" and "sources" in m:
                sources_html = '<div class="source-card"><strong>Sources:</strong> '
                sources_html += " | ".join(m["sources"])
                sources_html += '</div>'
                st.markdown(sources_html, unsafe_allow_html=True)

    # ---- Chat Input ----
    if question := st.chat_input("Ask anything — PDFs or the web..."):
        # Validate API keys
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
                    chat_history = build_chat_history()

                    result = None
                    for output in app.stream(
                        {"question": question, "chat_history": chat_history},
                        stream_mode="values"
                    ):
                        result = output

                    answer = result.get("answer", "I couldn't generate an answer. Please try again.")
                    documents = result.get("documents", [])
                    sender = result.get("sender", "")

                    st.markdown(answer)

                    # Extract sources for display
                    sources = []
                    for doc in documents:
                        src = doc.metadata.get("source", "")
                        if src and src not in sources and src != "error":
                            sources.append(src)

                    if sources:
                        sources_html = '<div class="source-card"><strong>Sources:</strong> '
                        source_items = []
                        for s in sources[:5]:  # Cap at 5 sources
                            if s.startswith("http"):
                                source_items.append(f'<a href="{s}" target="_blank" style="color:#7c3aed">{s[:60]}...</a>')
                            else:
                                source_items.append(s)
                        sources_html += " | ".join(source_items)
                        sources_html += '</div>'
                        st.markdown(sources_html, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


# --- Run ---
if __name__ == "__main__":
    main()
