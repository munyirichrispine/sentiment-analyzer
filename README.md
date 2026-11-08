# Sentiment Analyzer

A Natural Language Processing (NLP) project that classifies movie reviews as positive or negative.

## What it does
- Loads 2,000 labeled movie reviews from the NLTK movie_reviews dataset
- Converts text to numbers using TF-IDF
- Trains a Logistic Regression model with scikit-learn
- Tests on 400 reviews the model has never seen
- Lets you type your own review and get a prediction

## Results
Accuracy: 82.3% on the held-out test set (400 reviews).

## How to run
1. Install Python 3
2. Install the libraries:
   pip install scikit-learn pandas nltk
3. Download the dataset:
   python -m nltk.downloader movie_reviews
4. Run the program:
   python sentiment.py

## Tools used
Python, scikit-learn, NLTK
