# 📄 DocQuery — Chat With Your Documents

> Upload any PDF or text file. Ask questions in plain English. Get answers grounded in the actual document — with source passages shown so you can verify every claim.

Built with **FastAPI + FAISS + OpenAI**. Fully functional RAG (Retrieval-Augmented Generation) pipeline in under 300 lines of code.

---

## 🚀 Live Demo

> **[https://docquery-rag-production.up.railway.app](https://docquery-rag-production.up.railway.app)**

---

## What It Does

Most AI chatbots make things up. This one can't — it's only allowed to answer from the documents you give it. Every answer shows exactly which passage it came from.

**Use cases:**
- Chat with a legal contract, research paper, or manual
- Ask questions about internal company docs
- Build a knowledge base you can actually query

---

## Architecture

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
│  Build prompt: "Answer ONLY from these chunks"              │
│             ↓                                               │
│  GPT-4o-mini → grounded answer + source citations           │
│                                                             │
│  FastAPI layer: /upload  /query  /query/stream  /sources    │
│  Frontend: single-file vanilla JS (no build step)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Benchmark Results

Run against a 20-page technical document using the included `evaluate.py`:

| Metric | Result |
|---|---|
| Pass rate | Run `python evaluate.py --doc your_file.pdf` |
| Citation rate | Measured per run |
| Avg query latency | Measured per run |
| Hallucination rate | Tracked via must-not-say rules |

To run the benchmark yourself:
```bash
python evaluate.py --doc path/to/your/document.pdf --verbose
```

Results are saved to `eval_report_<docname>.json`.

---

## How It Works (RAG in plain English)

RAG = **Retrieval-Augmented Generation**. It's the most in-demand AI pattern in 2026.

```
You upload a PDF
    ↓
Split into small overlapping text chunks (~800 chars each)
    ↓
Each chunk is converted to a vector (list of numbers) by OpenAI's embedding model
    ↓
Vectors stored in a FAISS index (fast similarity search)
    ↓
You ask a question → question also gets embedded
    ↓
Find top 4 chunks whose vectors are most similar to the question
    ↓
Send those chunks + question to GPT-4o-mini: "Answer ONLY from this context"
    ↓
Get a grounded answer with source citations
```

**Why not just paste the whole doc into ChatGPT?**
- Documents can be bigger than the model's context window
- Retrieval is cheaper — 4 chunks per question vs. the whole doc every time
- You get auditable sources — you know exactly what produced the answer

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI (Python) |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Frontend | Vanilla HTML/CSS/JS (zero build step) |
| PDF Parsing | pypdf |

---

## Run It Locally

```bash
git clone https://github.com/YOUR_USERNAME/rag-chat-app
cd rag-chat-app

# Create virtual environment (optional but recommended)
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY=sk-your-key-here

# Start the server
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser. Upload a PDF. Start asking.

**Cost:** A 20-page PDF + a dozen questions costs under $0.01 with the models used.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a PDF, .txt, or .md file |
| `POST` | `/query` | Ask a question, get a grounded answer |
| `GET` | `/sources` | List uploaded documents |
| `POST` | `/reset` | Clear all documents |
| `GET` | `/health` | Server health check |

---

## Project Structure

```
rag-chat-app/
├── app.py          # FastAPI server — 4 endpoints, thin wrapper around rag.py
├── rag.py          # Core RAG pipeline — chunking, embedding, FAISS, retrieval, generation
├── static/
│   └── index.html  # Frontend UI — single file, no build step
├── requirements.txt
└── .env.example
```

---

## Deploy It (Free)

**Railway** or **Render** — both support FastAPI directly from GitHub:

1. Push this repo to GitHub
2. Connect repo to Railway/Render
3. Set `OPENAI_API_KEY` as an environment variable in their dashboard
4. Deploy — they handle the rest

Live link goes in your portfolio and LinkedIn.

---

## What I'd Build Next

- **Persistent storage** — swap in-memory FAISS for Chroma or Pinecone so documents survive server restarts
- **Hybrid search** — combine vector search with keyword search so exact terms (names, numbers) aren't missed
- **Multi-turn conversation** — track Q&A history so follow-up questions have context
- **Per-user isolation** — auth layer so each user only sees their own documents
- **Retrieval evaluation** — build a test set to measure whether the right chunk surfaces for each question

---

## Built By

Nate Green — AI systems builder.

This project is part of a portfolio demonstrating real-world AI application development using AI-assisted workflows.

[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)
