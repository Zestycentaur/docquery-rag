# 📄 DocQuery — Chat With Your Documents

![Live](https://img.shields.io/badge/status-live-brightgreen)
![Stack](https://img.shields.io/badge/stack-FastAPI%20%C2%B7%20FAISS%20%C2%B7%20OpenAI-blue)
![Deployed](https://img.shields.io/badge/deployed-Railway-blueviolet)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> Upload any PDF or text file. Ask questions in plain English. Get answers grounded in the actual document — with source passages shown so you can verify every claim.

**[🚀 Try the Live Demo →](https://docquery-rag-production.up.railway.app)**

---

![DocQuery Screenshot](assets/screenshot.png)

---

## 💡 The Problem It Solves

Most AI chatbots make things up. DocQuery can't — it's only allowed to answer from documents you give it. Every answer cites the exact passage it came from.

**Use cases:**
- Chat with a legal contract, research paper, or manual
- Ask questions about internal company docs
- Build a queryable knowledge base from any text

---

## ⚡ Quick Start

```bash
git clone https://github.com/Zestycentaur/docquery-rag
cd docquery-rag
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
uvicorn app:app --port 5000
```

Then open `http://localhost:5000` and upload a document.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DOCQUERY                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INGEST PIPELINE                                            │
│  ─────────────                                              │
│  Upload PDF/TXT → pypdf extract → overlapping chunks        │
│       (800 chars, 150 overlap)                              │
│             ↓                                               │
│  OpenAI text-embedding-3-small → float32 vectors            │
│             ↓                                               │
│  FAISS IndexFlatIP (cosine sim, normalized) ← stored        │
│                                                             │
│  QUERY PIPELINE                                             │
│  ──────────────                                             │
│  User question → embed (same model)                         │
│             ↓                                               │
│  FAISS search → top-4 most similar chunks                   │
│             ↓                                               │
│  Prompt: "Answer ONLY from these chunks"                    │
│             ↓                                               │
│  GPT-4o-mini → grounded answer + source citations           │
│                                                             │
│  FastAPI: /upload  /query  /query/stream  /sources  /reset  │
│  Frontend: vanilla JS, no build step required               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| API Server | FastAPI |
| Vector Store | FAISS (IndexFlatIP, cosine similarity) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini |
| Document Parsing | pypdf |
| Frontend | Vanilla JS (single file, no build step) |
| Deployment | Railway (Docker + GitHub auto-deploy) |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a PDF or .txt file |
| `POST` | `/query` | Ask a question, get a grounded answer |
| `GET` | `/query/stream` | Streaming response version |
| `GET` | `/sources` | List all uploaded documents |
| `POST` | `/reset` | Clear all documents and start fresh |

---

## 🚀 Deployment

Deployed on **Railway** with:
- Docker container (python:3.11-slim)
- Auto-deploy on every GitHub push
- Public HTTPS domain provisioned automatically
- OpenAI API key injected via environment variables

---

## ⚠️ Disclaimer

Built as a portfolio/demo project. Not intended for production use with sensitive documents.

---

*Built with AI-assisted development (OpenClaw + Claude Sonnet)*
