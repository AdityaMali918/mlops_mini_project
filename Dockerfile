FROM python:3.11.4-slim

WORKDIR /app

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

COPY flask_app/ /app/

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

# Start the application
# CMD ["gunicorn","-b", "0.0.0.0:5000", "app:app"]
CMD ["gunicorn", "--preload", "-w", "2", "--timeout", "120", "-b", "0.0.0.0:5000", "app:app"]