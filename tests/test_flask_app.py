from flask_app.app import app


def test_home_page():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"<title>Sentiment Analysis</title>" in response.data


def test_predict_page():
    client = app.test_client()

    response = client.post(
        "/predict",
        data={"text": "I love this!"}
    )

    assert response.status_code == 200
    assert (
        b"Happy" in response.data or
        b"Sad" in response.data
    )
