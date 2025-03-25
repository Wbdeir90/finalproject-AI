import pytest
from api.app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json == {"status": "ok"}

def test_spam_detection(client):
    data = {"message": "Win $1000 now!"}
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    assert "prediction" in response.json
