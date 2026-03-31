from fastapi.testclient import TestClient

import api

client = TestClient(api.app)

def test_home():
    """Test de l'endpoint '/'"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "L'API marche correctement"}

def test_analyse_sentiment_positive():
    """Test texte positif"""
    data = {"text": "I Love this app !"} # Fonctionne mieux en anglais
    response = client.post("/analyse_sentiment/", json=data)
    assert response.status_code == 200
    json_data = response.json()
    for key in ["neg", "neu", "pos", "compound"]:
        assert key in json_data
    # Le score positif > score négatif
    assert json_data["pos"] > json_data["neg"]

def test_analyse_sentiment_negative():
    """Test texte négatif"""
    data = {"text": "I hate this app !"}
    response = client.post("/analyse_sentiment/", json=data)
    assert response.status_code == 200
    json_data = response.json()
    for key in ["neg", "neu", "pos", "compound"]:
        assert key in json_data
    # Le score négatif > score positif
    assert json_data["neg"] > json_data["pos"]

def test_analyse_sentiment_neutre():
    """Test texte neutre"""
    data = {"text": "I feel normal."}
    response = client.post("/analyse_sentiment/", json=data)
    assert response.status_code == 200
    json_data = response.json()
    for key in ["neg", "neu", "pos", "compound"]:
        assert key in json_data
    # Le score neutre > score positif et score négatif
    assert json_data["neu"] > json_data["pos"]
    assert json_data["neu"] > json_data["neg"]