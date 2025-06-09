import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("Recipe Reviews and User Feedback Dataset.csv")
    df = df[['text', 'stars']].dropna()
    df = df[df['stars'].isin([1, 2, 3, 4, 5])]
    df['label'] = df['stars'].apply(lambda x: 1 if x >= 4 else 0)
    return df

# Train model with caching
@st.cache_resource
def train_model(data):
    X = data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    return pipeline, accuracy

# Streamlit App UI
st.set_page_config(page_title="Recipe Review Classifier", layout="centered")
st.title("🥘 Recipe Review Sentiment Classifier")
st.markdown("This app uses **Logistic Regression** to predict whether a recipe review is **Positive (👍)** or **Negative (👎)**.")

# Load and train
data = load_data()
model, acc = train_model(data)

# Show accuracy
st.write(f"✅ Model accuracy: **{acc:.2%}**")

# 📊 Show distribution of stars
st.subheader("⭐ Distribution of Review Ratings")
fig, ax = plt.subplots()
sns.countplot(data=data, x='stars', palette='viridis', ax=ax)
ax.set_title("Number of Reviews by Star Rating")
ax.set_xlabel("Star Rating")
ax.set_ylabel("Count")
st.pyplot(fig)

# Text input
st.subheader("🔍 Predict Sentiment from Review")
user_input = st.text_area("Enter a review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])[0]
        label = "👍 Positive Review" if prediction == 1 else "👎 Negative Review"
        st.success(f"Prediction: {label}")
