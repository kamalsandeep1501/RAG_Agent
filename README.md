# 🤖 RAG Agent with PDF Upload & Web Search

An intelligent AI-powered chatbot that can answer questions using both **uploaded PDF documents** and **real-time web data**. This project uses **Retrieval-Augmented Generation (RAG)** along with an **AI routing system** to provide accurate and context-aware responses.

---

## 🚀 Live Demo

🌐 **Live Deploy:**  
[RAG Agent Live Demo](https://ragagent-avduawyfenilbybxmemvwx.streamlit.app/)

---

## 🚀 Features

- 📤 Upload and analyze PDF documents  
- 🧠 Smart AI routing (decides between PDF or web search)  
- 🌐 Real-time web search (WHO-based trusted sources)  
- 💬 Interactive chat interface  
- 📚 Source-based answers (PDF or URL)  
- ⚡ Fast and efficient response generation  

---

## 🧠 Project Overview

Traditional chatbots rely only on pre-trained knowledge and cannot access custom documents or live data.

This project solves that by:
- Reading **user-uploaded PDFs**
- Fetching **real-time web information**
- Using **RAG (Retrieval-Augmented Generation)** to generate accurate answers

---

## 🔄 Workflow

```text
User uploads PDF
        ↓
PDF is processed and split into chunks
        ↓
Chunks converted into embeddings
        ↓
Stored in Chroma Vector Database
        ↓
User asks question
        ↓
AI Router decides:
   → Use PDF (vectorstore)
   → OR Web Search
        ↓
Relevant data retrieved
        ↓
LLM generates final answer
        ↓
Answer displayed with source
```

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **LLM Framework:** LangChain  
- **Vector Database:** ChromaDB  
- **Embeddings:** HuggingFace Embeddings  
- **LLM:** Groq / Gemini / OpenAI  
- **PDF Processing:** PyPDFLoader  
- **Web Search:** Tavily Search API  
- **Environment Management:** Python Dotenv  

---

## 📌 Use Cases

- 📚 Research Assistant  
- 🏥 Healthcare Information Retrieval  
- 📄 Company Policy Q&A  
- 🎓 Educational Chatbot  
- 📑 Document-based AI Assistant  

---

## 💡 Key Concepts Used

- Retrieval-Augmented Generation (RAG)  
- Vector Embeddings  
- Semantic Search  
- AI Routing Logic  
- Prompt Engineering  
- Context-Aware Response Generation  

---

## 📷 Project Output

- Users can upload PDFs and ask questions directly from the document  
- If the answer is unavailable in the PDF, the agent automatically switches to web search  
- Responses include the source of information for better reliability  

---

## 🔮 Future Improvements

- Multi-PDF support  
- Chat history memory  
- Voice-based interaction  
- Authentication system  
- Advanced reranking for retrieval accuracy  
- Deployment using Docker & Kubernetes  

---

## 👨‍💻 Author

Developed by **Kamal Sandeep** 🚀
