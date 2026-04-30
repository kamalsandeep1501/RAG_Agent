# 🤖 RAG Agent with PDF Upload & Web Search

An intelligent AI-powered chatbot that can answer questions using both **uploaded PDF documents** and **real-time web data**. This project uses **Retrieval-Augmented Generation (RAG)** along with an **AI routing system** to provide accurate and context-aware responses.

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
