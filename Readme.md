# 📰 Fake News Detection using Machine Learning

This project is a machine learning-based web app that detects whether a news article is **real** or **fake** using **Natural Language Processing (NLP)** and a **Naive Bayes classifier**. It features an interactive UI built with **Streamlit** and runs completely in the browser.

## 🚀 Live Demo
🔗 [Click here to try the app](https://fake-news-detection-n6chhhyxzcqlhcymozdzz2.streamlit.app/)

---

## 📌 Features

- Text cleaning with NLTK
- TF-IDF vectorization
- Naive Bayes model training
- Accuracy, confusion matrix, and classification report
- Real-time prediction with a clean web UI

---

## 🧠 Technologies Used

- Python 3.x
- Streamlit
- Pandas
- scikit-learn
- NLTK
- Joblib

---

## 📂 Project Structure

fake-news-detection/
├── app/
│ └── app.py # Streamlit web app
├── data/
│ ├── Fake.csv # Fake news dataset
│ └── True.csv # Real news dataset
├── models/
│ ├── model_nb.pkl # Trained model
│ └── vectorizer.pkl # Saved TF-IDF vectorizer
├── utils/
│ └── text_cleaning.py # Custom text preprocessing
├── load_data.py # Data loading, preprocessing, model training
├── requirements.txt # Python dependencies
└── README.md # Project overview


---

## 🧪 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection

2.Install dependencies:
pip install -r requirements.txt

3.Train the model:
python load_data.py

4.Launch the app:
streamlit run app/app.py

📊 Sample Output
Model Accuracy: ~96%

Real-time predictions: 🟢 Real News / 🔴 Fake News

Clean text input field and responsive design

📚 Dataset Source
Kaggle Fake and Real News Dataset


👤 Author
Your Name: Manoj Kumar

Email: manojkumar112128@gmail.com

GitHub: Manoj-2107


✅ Future Enhancements
Add WordCloud of most common words

Include more models like Logistic Regression or SVM

Deploy via Hugging Face or Docker

