from unittest.mock import Mock

import requests
from streamlit.testing.v1 import AppTest


def test_empty_text_warning():
    at = AppTest.from_file("app.py")
    at.run()

    at.text_area[0].set_value(" ")
    at.button[0].click()
    at.run()

    assert "Veuillez saisir un texte" in at.warning[0].value

def test_sentiment_analysis_success():
    at = AppTest.from_file("app.py")
    at.run()

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "neg": 0.0,
        "neu": 0.2,
        "pos": 0.8,
        "compound": 0.7
    }

    requests.post = Mock(return_value=mock_response)

    at.text_area[0].set_value("I love this app")
    at.button[0].click()
    at.run()

    assert "Sentiment : Sentiment global : Positif" in at.success[0].value
    assert "Score global" in at.metric[0].label
    assert "Détails" in at.subheader[0].value