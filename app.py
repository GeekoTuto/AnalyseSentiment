import requests
import streamlit as st


API_URL = "http://127.0.0.1:8001/analyse_sentiment/"


def get_sentiment_text(compound_score: float) -> str:
	if compound_score >= 0.05:
		return "Sentiment global : Positif 😀"
	if compound_score <= -0.05:
		return "Sentiment global : Négatif 🙁"
	return "Sentiment global : Neutre 😐"


st.set_page_config(page_title="Analyse de sentiment", layout="centered")

st.title("Analyse de sentiment")
st.write("Entrez un texte")

api_url = st.text_input("URL API", value=API_URL)
text = st.text_area("Texte a analyser", height=150, placeholder="Exemple : I love this App !")

if st.button("Analyser", type="secondary"):
	if not text.strip():
		st.warning("Veuillez saisir un texte avant de lancer l'analyse.")
	else:
		try:
			response = requests.post(api_url, json={"text": text})
			response.raise_for_status()
			result = response.json()
			sentiment_text = get_sentiment_text(result["compound"])

			st.success(f"Sentiment : {sentiment_text}")
			st.metric("Score global", f"{result['compound']:.4f}")
			st.subheader("Details")
			st.json(result)
		except requests.exceptions.ConnectionError:
			st.error("Connexion impossible a l'API")
		except requests.exceptions.HTTPError as exc:
			st.error(f"Erreur HTTP : {exc}")
		except requests.exceptions.RequestException as exc:
			st.error(f"Erreur lors de l'appel API : {exc}")
