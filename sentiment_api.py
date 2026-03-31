import nltk
nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

sentiment = sia.polarity_scores("bonjour") 
print(sentiment)