import pytest
from app.api import app as flask_app

@pytest.fixture
def client():
    flask_app.testing = True
    return flask_app.test_client()

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200

def test_predict_no_file(client):
    r = client.post("/predict", data={})
    assert r.status_code == 400