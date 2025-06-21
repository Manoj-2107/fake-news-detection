# load_data.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.text_cleaning import clean_text

print("Script started...")

# Load CSV files
try:
    fake_df = pd.read_csv("data/Fake.csv")
    print("Loaded Fake.csv")
    real_df = pd.read_csv("data/True.csv")
    print("Loaded True.csv")
except Exception as e:
    print("Error loading data:", e)
    exit()

# Label and merge
fake_df['label'] = 0
real_df['label'] = 1
df = pd.concat([fake_df, real_df], axis=0).sample(frac=1).reset_index(drop=True)
print("Combined dataset shape:", df.shape)

# Clean text
df['text'] = df['title'] + " " + df['text']
df['text'] = df['text'].apply(clean_text)
print("Cleaned text column created")

# Split and vectorize
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("TF-IDF Training Matrix Shape:", X_train_vec.shape)
print("TF-IDF Test Matrix Shape:", X_test_vec.shape)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, "models/model_nb.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("\nModel and vectorizer saved to /models/")
