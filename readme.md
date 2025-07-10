# RAG Chatbot: Retrieval-Augmented Generation Chatbot

A simple and modular Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **OpenAI**, **ChromaDB**, and **Gradio**. This project demonstrates how to combine a local document knowledge base with an LLM to build a context-aware question-answering system.

---

## Features

- ✅ Upload and embed your own documents
- 🔍 Retrieves relevant chunks based on user query
- 💬 Answers questions using OpenAI (GPT-3.5+)
- 🛢 Vector database powered by Chroma
- ⚙️ Modular design using LangChain
- 🎨 Simple Gradio front end

---

## Project Structure
```
RAG-chatbot/
├── app/
│ ├── ingest.py # Load and index documents into ChromaDB
│ ├── chatbot.py # Set up retriever + LLM pipeline (LangChain RetrievalQA)
│ ├── gradio_app.py # Gradio web interface
│ └── data/
│ └── example.txt # Sample source document
├── db/ # Persisted ChromaDB vectorstore
├── .env # (Optional) API keys
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/RAG-chatbot.git
cd RAG-chatbot

conda create -n rag-chatbot python=3.10
conda activate rag-chatbot

pip install -r requirements.txt
```
### 2. Add OpenAI Key
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Ingest Documents
```bash
python app/ingest.py
```

### 4. Run the App
```bash
python app/gradio_app.py
```
Go to http://localhost:7860 to interact with the chatbot