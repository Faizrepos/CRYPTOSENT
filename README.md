# CRYPTOSENT: Airline Sentiment & Emotion Analysis Project

This project, titled **CRYPTOSENT**, is a complete sentiment analysis and emotion detection pipeline built using Python. It processes and classifies airline-related text data to identify **positive**, **negative**, or **neutral** sentiments and detect associated **emotions** using the NRC Emotion Lexicon.

> ⚠️ **Note**: The main script provided in this repository contains file paths specific to the creator’s local development environment. This project is strictly for **viewing and educational purposes only**, and not intended for direct usage or deployment on other systems.

---

## 🧠 Features

- Automatic dataset extraction from ZIP
- Full text preprocessing pipeline (tokenization, stopword removal, lemmatization)
- TF-IDF vectorization of cleaned text
- Logistic Regression model for sentiment classification
- Visualization of:
  - Sentiment distribution (donut pie chart)
  - Word count transformation during preprocessing
  - Emotion distribution using NRC Emotion Lexicon
  - Confusion matrix heatmap
- Classification report and accuracy metrics

---

## 🛠️ Tech Stack & Libraries

- **Language**: Python 3.x
- **Libraries Used**:
  - `pandas`, `numpy` – data manipulation
  - `matplotlib`, `seaborn` – data visualization
  - `nltk` – natural language processing (tokenization, stopwords, lemmatization)
  - `sklearn` – TF-IDF vectorization, model building, and evaluation
  - `zipfile`, `os`, `re` – file handling and text cleaning
- **Lexicons**:
  - NRC Emotion Lexicon (for emotion tagging)

---


## 🗂️ Project Structure

```plaintext
Cryptosent/
├── main_project.py                # Main Python file for the project
├── datasets/
│   └── dataset.csv       # dataset file
└── essential libs/
    └── NRC-Emotion-Lexicon/
       ├── NRC-Emotion-Lexicon-Wordlevel-v0.92.txt  # Example essential file
       └── ...                                       # Other necessary resources



---

## 📌 Disclaimer

- The project is **NOT** meant to be run as-is on other systems due to local file path dependencies.
- incase of accessing this project, User should Custom path handling or refactoring is required to use this on another machine.

---

## 💡 Author Note

This project was created by **Faiz** as part of a larger initiative to build real-world sentiment analysis tools with NLP and machine learning.

---

