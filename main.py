import streamlit as st
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import re
import nltk
# nltk.download()
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Set the page title
st.title("Simple GUI Program in NLP")

# Add a text element to the page
st.write("Hello, viewer!")

# Add an interactive input field
review = st.text_input("Enter your review", "")
model_choice = st.selectbox("Choose Model", ("CLF","SVC", "Naive"))
# Add a button
button = st.button("Submit")


def extact_features(text):
    cleaded = re.sub(r'[^\w\s]', '', text)
    cleaned_text = cleaded.lower()
    tokens = word_tokenize(cleaned_text)
    # Get the list of stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords from the tokenized text
    filtered_token = [token for token in tokens if token.lower() not in stop_words]

    joined_string = ' '.join(filtered_token)
    return joined_string

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

review = vectorizer.transform([extact_features(review)])

def predict(text_features, model_type):
    if (model_type == "Naive"):
        naive = pickle.load(open('Naive.pkl', 'rb'))
        y_pred = pd.DataFrame(naive.predict_proba(text_features))

    elif model_type == "CLF":
        DecisionTree = pickle.load(open('MLP.pkl', 'rb'))
        y_pred = pd.DataFrame(DecisionTree.predict_proba(text_features))

    elif model_type == "SVC":
        SVC = pickle.load(open('SVC.pkl', 'rb'))
        y_pred = pd.DataFrame(SVC.predict_proba(text_features))

    return "Negative : {} \n\n  Positive : {}".format(y_pred.iat[0, 0], y_pred.iat[0, 1])


# Display a message when the button is clicked
if button:
    st.write(predict(review,model_choice))

