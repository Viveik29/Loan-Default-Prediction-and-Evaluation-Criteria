from app import create_app

def test_home_page():
    app = create_app(testing=True)
    client = app.test_client()

    res = client.get("/")
    assert res.status_code == 200


def test_predict_endpoint_without_model():
    app = create_app(testing=True)
    client = app.test_client()

    res = client.post("/predict", json={"x": 1})
    assert res.status_code == 500
