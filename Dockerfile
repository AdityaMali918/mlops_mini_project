# Builder
FROM python:3.11.4-slim AS builder

WORKDIR /app

COPY flask_app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet

# Runtime
FROM python:3.11.4-slim

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY flask_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

CMD ["gunicorn", "--preload", "-w", "2", "--timeout", "120", "-b", "0.0.0.0:5000", "app:app"]