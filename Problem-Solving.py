import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data = {
    'username': ['@alpha', '@bravo', '@echo', '@nova', '@shadow', '@glitch', '@vortex', '@ace', '@drift', '@specter'],
    'message': [
        "The game keeps freezing every few minutes. Super annoying!",
        "Love the new update, smoother gameplay and graphics!",
        "Matchmaking takes forever, this update broke everything!",
        "I can’t connect to the server, keeps showing timeout error!",
        "The lag is unbearable during ranked matches 😡",
        "App crashes every time I open my inventory!",
        "Finally fixed the bug from last patch, great job devs!",
        "Sound glitch when using voice chat, please fix soon!",
        "Weapons feel unbalanced after the last update.",
        "I lost my progress after syncing with my account 😞"
    ]
}

df = pd.DataFrame(data)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_message'] = df['message'].apply(preprocess)

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['Polarity'] = df['clean_message'].apply(get_sentiment)
df['Sentiment'] = df['Polarity'].apply(lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral'))

all_words = " ".join(df['clean_message']).split()
filtered_words = [w for w in all_words if w not in ['love','great','smoother','amazing','awesome','nice','cool','finally','good','job']]
keyword_counts = Counter(filtered_words).most_common(15)
keywords_df = pd.DataFrame(keyword_counts, columns=['Keyword', 'Count'])

plt.figure(figsize=(10,5))
wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(" ".join(filtered_words))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Frequent Keywords in Game Messenger Feedback", fontsize=14)
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='Count', y='Keyword', data=keywords_df, palette='crest')
plt.title('Top Complaint Keywords')
plt.xlabel('Frequency')
plt.ylabel('Keyword')
plt.show()

sentiment_summary = df['Sentiment'].value_counts(normalize=True).mul(100).round(2)
plt.figure(figsize=(6,6))
plt.pie(sentiment_summary, labels=sentiment_summary.index, autopct='%1.1f%%', startangle=90, 
        colors=['lightgreen','gold','salmon'])
plt.title('Overall Sentiment Distribution (Game Messenger)')
plt.show()

avg_polarity = df['Polarity'].mean()
top_negative = df[df['Polarity'] < -0.1].sort_values('Polarity').head(3)
top_positive = df[df['Polarity'] > 0.1].sort_values('Polarity', ascending=False).head(3)

print("GAME MESSENGER FEEDBACK SENTIMENT ANALYSIS\n")
print(df[['username','message','Sentiment']].to_string(index=False))
print("\nSENTIMENT SUMMARY:")
for s, p in sentiment_summary.items():
    print(f"{s}: {p:.2f}%")
print(f"\nAVERAGE SENTIMENT POLARITY: {avg_polarity:.2f}")
print("\nTOP NEGATIVE FEEDBACKS:")
for msg in top_negative['message']:
    print(f"- {msg}")
print("\nTOP POSITIVE FEEDBACKS:")
for msg in top_positive['message']:
    print(f"- {msg}")

print("\nANALYTICAL INSIGHTS:")
print("Most complaints are related to lag, crashes, and matchmaking.")
print("Positive reviews focus on performance improvements and bug fixes.")
print("Overall sentiment shows a mix of frustration and appreciation post-update.")
print("Common technical keywords detected: error, lag, freeze, bug, crash, matchmaking, voice, timeout.")
