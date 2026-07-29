# --------------- Builder Stage ------------------
FROM python:3.11.4-slim AS builder

WORKDIR /app

# Install build dependencies if required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY flask_app/requirements.txt .

# Install Python dependencies into a separate location
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader -d /install/nltk_data stopwords wordnet

# ---------- Runtime Stage ----------
FROM python:3.11.4-slim

WORKDIR /app

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Make NLTK data available
ENV NLTK_DATA=/usr/local/nltk_data

# Copy application
COPY flask_app/ /app/

# Copy model
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

EXPOSE 5000

CMD ["gunicorn", "--preload", "-w", "2", "--timeout", "120", "-b", "0.0.0.0:5000", "app:app"]