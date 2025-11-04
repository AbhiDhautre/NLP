# Bike Review Intelligence System

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
import numpy as np


reviews = [
    "The Royal Enfield Classic 350 is smooth and powerful but a bit heavy for city rides.",
    "Yamaha R15 has great mileage and stunning looks. Absolutely love it!",
    "Hero Splendor is perfect for daily commute. Very reliable and fuel-efficient.",
    "Bajaj Pulsar gives amazing pickup but engine noise increases over time.",
    "Honda Shine offers good comfort and a smooth ride experience.",
    "TVS Apache RTR 160 feels sporty but the seat is not comfortable for long rides.",
    "Suzuki Gixxer’s performance is good but spare parts are costly.",
    "Royal Enfield Himalayan is great for touring. Excellent comfort on rough roads.",
    "KTM Duke 200 is stylish and fast but maintenance cost is high.",
    "Bajaj Avenger gives a cruiser feel and good comfort for long-distance travel."
]


vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(reviews)

query = input("🔎 Enter your search query (e.g., comfort, mileage, performance, cost): ")
query_vec = vectorizer.transform([query])
similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
ranked_indices = similarity_scores.argsort()[::-1]


analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

rake = Rake()

def extract_keywords(text):
    rake.extract_keywords_from_text(text)
    return ', '.join(rake.get_ranked_phrases()[:3])  


print("\n🏍️ --- Search Results ---\n")
results = []
for i in ranked_indices:
    if similarity_scores[i] > 0:
        sentiment = get_sentiment(reviews[i])
        keywords = extract_keywords(reviews[i])
        results.append({
            "Score": round(similarity_scores[i], 3),
            "Review": reviews[i],
            "Sentiment": sentiment,
            "Keywords": keywords
        })
        print(f"Score: {similarity_scores[i]:.3f}")
        print(f"Review: {reviews[i]}")
        print(f"Sentiment: {sentiment}")
        print(f"Keywords: {keywords}")
        print("-" * 80)

df = pd.DataFrame(results)
print("\n📊 --- Summary Report ---\n")
if not df.empty:
    print("Total Reviews Retrieved:", len(df))
    print(df['Sentiment'].value_counts())
    print("\nMost Common Keywords:")
    all_keywords = ', '.join(df['Keywords'].tolist()).split(', ')
    keywords_series = pd.Series(all_keywords)
    print(keywords_series.value_counts().head(5))
else:
    print("No relevant reviews found for your query.")
    comfort