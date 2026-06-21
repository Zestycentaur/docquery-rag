FROM python:3.11-slim

WORKDIR /app

# Install system deps for pypdf and faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway/Render inject PORT as an env var; default to 8000 locally
ENV PORT=8000

EXPOSE $PORT

CMD uvicorn app:app --host 0.0.0.0 --port $PORT
