import streamlit as st
import pickle

# Load vectorizer and model separately
with open("tfidf_model.pkl", "rb") as f:
    vectorizer, classifier = pickle.load(f)

st.title("AI vs Human Assignment Detector")

st.write("Enter your assignment text below:")

user_text = st.text_area("Text input")

if st.button("Check"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        text_vector = vectorizer.transform([user_text])
        prediction = classifier.predict(text_vector)[0]

        if prediction == "AI-generated":
            st.error("This text is AI-generated ❌")
        else:
            st.success("This text is Human-written ✅")

