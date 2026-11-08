from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

texts = []
labels = []
for category in movie_reviews.categories():
    for file_id in movie_reviews.fileids(category):
        texts.append(movie_reviews.raw(file_id))
        labels.append(category)

print("Total reviews loaded:", len(texts))

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

model = make_pipeline(
    TfidfVectorizer(stop_words="english", max_features=20000),
    LogisticRegression(max_iter=1000),
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", round(model.score(X_test, y_test), 3))
print(classification_report(y_test, predictions))

while True:
    text = input("Type a movie review sentence (or 'quit'): ")
    if text.lower() == "quit":
        break
    print("->", model.predict([text])[0])