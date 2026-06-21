"""
app.py — FastAPI server.

Three endpoints, that's the whole API:
  POST /upload   -> ingest a document (PDF or .txt)
  POST /query    -> ask a question, get an answer grounded in uploaded docs
  GET  /sources  -> list what's been uploaded so far
  POST /reset    -> clear the store (start fresh)

Run with:  uvicorn app:app --reload --port 8000
Then open static/index.html in a browser (it talks to localhost:8000).
"""

import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIError
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag

app = FastAPI(title="Chat With Your Docs — RAG demo")

# CORS wide open since this is a local demo, not a multi-tenant production app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    history: list[dict] | None = None  # [{role, content}, ...] for multi-turn conversation


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(400, "Only .pdf, .txt, or .md files are supported.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        num_chunks = await run_in_threadpool(rag.ingest_document, tmp_path, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except APIError as e:
        raise HTTPException(502, f"OpenAI API error: {e.message}")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error during ingestion: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return {"filename": file.filename, "chunks_added": num_chunks}


@app.post("/query")
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    try:
        result = await run_in_threadpool(rag.answer_question, req.question, req.history)
    except APIError as e:
        raise HTTPException(502, f"OpenAI API error: {e.message}")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error during query: {str(e)}")
    return result


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming version of /query. Emits server-sent events (SSE).

    The frontend reads these as a stream and appends tokens as they arrive,
    giving the "typing" effect that makes AI responses feel live.
    Format: each chunk is a JSON line: {"token": "..."}  or  {"sources": [...]}
    """
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    if rag.STORE.is_empty():
        async def empty_gen():
            yield 'data: {"token": "No documents uploaded yet. Upload a PDF or text file first."}\n\n'
            yield 'data: {"done": true}\n\n'
        return StreamingResponse(empty_gen(), media_type="text/event-stream")

    # Retrieve sources synchronously (fast — just vector math)
    try:
        query_vector = await run_in_threadpool(rag.embed_texts, [req.question])
        results = await run_in_threadpool(rag.STORE.search, query_vector)
    except APIError as e:
        raise HTTPException(502, f"OpenAI API error: {e.message}")

    context_blocks = []
    sources = []
    for chunk, score in results:
        context_blocks.append(f"[Source: {chunk.source}, chunk {chunk.chunk_id}]\n{chunk.text}")
        sources.append({
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "score": round(score, 3),
            "preview": chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
        })

    context = "\n\n---\n\n".join(context_blocks)
    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the provided context. "
        "If the context doesn't contain the answer, say so clearly instead of guessing. "
        "Cite which source/chunk you used when relevant."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.question}\n\nAnswer based only on the context above."

    import json as _json

    async def token_gen():
        try:
            stream = rag.client.chat.completions.create(
                model=rag.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {_json.dumps({'token': delta})}\n\n"
            # Send sources at the end
            yield f"data: {_json.dumps({'sources': sources})}\n\n"
            yield 'data: {"done": true}\n\n'
        except APIError as e:
            yield f'data: {{"error": "OpenAI API error: {e.message}"}}\n\n'

    return StreamingResponse(token_gen(), media_type="text/event-stream")


@app.get("/sources")
async def sources():
    return {"sources": rag.STORE.list_sources(), "total_chunks": len(rag.STORE.chunks)}


@app.post("/reset")
async def reset():
    rag.reset_store()
    return {"status": "cleared"}


@app.get("/health")
async def health():
    return {"status": "ok"}


# Serve the frontend at the root URL too, so you can just open localhost:8000
app.mount("/", StaticFiles(directory="static", html=True), name="static")
