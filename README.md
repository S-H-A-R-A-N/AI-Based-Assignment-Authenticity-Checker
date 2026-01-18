# AI-Based Assignment Authenticity Checker

This project is an AI-powered system designed to detect whether an academic assignment is written by a human or generated using AI tools. It uses Natural Language Processing (NLP) and Machine Learning techniques to analyze text patterns and classify the input as either "AI-generated" or "Human-written". The final model is deployed as a web application using Streamlit.

---

## 📌 Features
- Detects AI-generated text vs human-written text
- Uses NLP for text preprocessing
- Trains a machine learning model
- Provides a user-friendly web interface using Streamlit

---

## 🧠 Technologies Used
- Python
- NLTK (Natural Language Toolkit)
- Scikit-learn
- Streamlit
- Pandas & NumPy

---

## 📁 Project Structure

Fake_Assignment-Detection/
│
├── ai_vs_human_text.csv # Dataset
├── model.py # Model training code
├── app.py # Streamlit app
├── requirements.txt # Required libraries
├── README.md # Project documentation
└── tfidf_model.pkl # Saved trained model (optional)

Install required libraries
pip install -r requirements.txt

How to Run
✅ Step 1: Train the model
python model.py

This will:
Clean the text using NLTK
Convert text to TF-IDF features
Train a Naive Bayes model
Save the model as tfidf_model.pkl

✅ Step 2: Run the Streamlit app
py -m streamlit run app.py

Example
Input:
This assignment was a bit difficult for me because I missed one class. I searched for information on Google and used my class notes to complete it. There might be some mistakes, but I have written everything based on my understanding.

Output:
Human-written
