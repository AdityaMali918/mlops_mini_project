# from flask import Flask, render_template,request
# import mlflow
# import pickle
# import os
# import pandas as pd

# import numpy as np
# import pandas as pd
# import os
# import re
# import nltk
# import string
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# app = Flask(__name__)

# #load model 
# from dotenv import load_dotenv

# load_dotenv()

# # mlflow.set_tracking_uri('http://127.0.0.1:5000/')

# dagshub_token = os.getenv("DAGSHUB_PAT")
# if not dagshub_token:
#         raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "AdityaMali918"
# repo_name = "mlops_mini_project"
# REGISTERED_MODEL_NAME = "my_model"
# ALIAS = "development"

# # Loads whatever version currently holds the "champion" alias —
# # register_model.py is what controls which version that is.
# model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{ALIAS}")
# vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

# def lemmatization(text):
#     """Lemmatize the text."""
#     lemmatizer = WordNetLemmatizer()
#     text = text.split()
#     text = [lemmatizer.lemmatize(word) for word in text]
#     return " ".join(text)

# def remove_stop_words(text):
#     """Remove stop words from the text."""
#     stop_words = set(stopwords.words("english"))
#     text = [word for word in str(text).split() if word not in stop_words]
#     return " ".join(text)

# def removing_numbers(text):
#     """Remove numbers from the text."""
#     text = ''.join([char for char in text if not char.isdigit()])
#     return text

# def lower_case(text):
#     """Convert text to lower case."""
#     text = text.split()
#     text = [word.lower() for word in text]
#     return " ".join(text)

# def removing_punctuations(text):
#     """Remove punctuations from the text."""
#     text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
#     text = text.replace('؛', "")
#     text = re.sub('\s+', ' ', text).strip()
#     return text

# def removing_urls(text):
#     """Remove URLs from the text."""
#     url_pattern = re.compile(r'https?://\S+|www\.\S+')
#     return url_pattern.sub(r'', text)

# def remove_small_sentences(df):
#     """Remove sentences with less than 3 words."""
#     for i in range(len(df)):
#         if len(df.text.iloc[i].split()) < 3:
#             df.text.iloc[i] = np.nan

# def normalize_text(text):
#     text = lower_case(text)
#     text = remove_stop_words(text)
#     text = removing_numbers(text)
#     text = removing_punctuations(text)
#     text = removing_urls(text)
#     text = lemmatization(text)

#     return text

# @app.route("/")
# def home():
#     print("HI")
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     text = request.form['text']

#     text = request.form['text']

#     # clean
#     text = normalize_text(text)

#     # bow
#     features = vectorizer.transform([text])

#     # Convert sparse matrix to DataFrame
#     features_df = pd.DataFrame.sparse.from_spmatrix(features)
#     features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

#     # prediction
#     result = model.predict(features_df)

#     return render_template('index.html', result=result[0])

# if __name__ == "__main__":
#     app.run(port=8000,debug=True)    


from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv


# -----------------------------
# Flask initialization
# -----------------------------
app = Flask(__name__)


# -----------------------------
# Environment configuration
# -----------------------------
load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_PAT environment variable is not set"
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


# -----------------------------
# MLflow configuration
# -----------------------------
DAGSHUB_URL = "https://dagshub.com"
REPO_OWNER = "AdityaMali918"
REPO_NAME = "mlops_mini_project"

mlflow.set_tracking_uri(
    f"{DAGSHUB_URL}/{REPO_OWNER}/{REPO_NAME}.mlflow"
)


REGISTERED_MODEL_NAME = "my_model"

# production for deployed app
# development for testing
MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "production"
)


# -----------------------------
# Lazy loaded artifacts
# -----------------------------
model = None
vectorizer = None


def load_artifacts():
    """
    Load MLflow model and vectorizer only when prediction is requested.
    """

    global model
    global vectorizer

    if model is None:
        print("Loading MLflow model...")

        model = mlflow.sklearn.load_model(
            f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        )

    if vectorizer is None:
        print("Loading vectorizer...")

        with open(
            "models/vectorizer.pkl",
            "rb"
        ) as file:
            vectorizer = pickle.load(file)

    return model, vectorizer



# -----------------------------
# Text preprocessing functions
# -----------------------------
def lemmatization(text):

    lemmatizer = WordNetLemmatizer()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)



def remove_stop_words(text):

    stop_words = set(
        stopwords.words("english")
    )

    words = [
        word
        for word in str(text).split()
        if word not in stop_words
    ]

    return " ".join(words)



def removing_numbers(text):

    return "".join(
        [
            char
            for char in text
            if not char.isdigit()
        ]
    )



def lower_case(text):

    words = text.split()

    words = [
        word.lower()
        for word in words
    ]

    return " ".join(words)



def removing_punctuations(text):

    text = re.sub(
        '[%s]' % re.escape(string.punctuation),
        ' ',
        text
    )

    text = text.replace('؛', "")

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text



def removing_urls(text):

    url_pattern = re.compile(
        r'https?://\S+|www\.\S+'
    )

    return url_pattern.sub(
        '',
        text
    )



def normalize_text(text):

    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text



# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():

    return render_template(
        "index.html",result=None
    )



@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    model, vectorizer = load_artifacts()

    text = request.form["text"]


    # preprocess
    text = normalize_text(text)


    # vectorize
    features = vectorizer.transform(
        [text]
    )


    features_df = pd.DataFrame(
        features.toarray(),
        columns=[
            str(i)
            for i in range(
                features.shape[1]
            )
        ]
    )


    # prediction
    result = model.predict(
        features_df
    )


    return render_template(
        "index.html",
        result=result[0]
    )



# -----------------------------
# Run application
# -----------------------------
if __name__ == "__main__":

    app.run(
        port=5000,
        host="0.0.0.0",
    )