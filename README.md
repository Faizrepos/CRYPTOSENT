# ✨ CRYPTOSENT: Airline Sentiment & Emotion Analysis Project

This project, titled **CRYPTOSENT**, is a complete sentiment analysis and emotion detection pipeline built using Python. It processes and classifies airline-related text data to identify **positive**, **negative**, or **neutral** sentiments and detect associated **emotions** using the NRC Emotion Lexicon.

> ⚠️ **Important Note**: The main script in this repository uses file paths hardcoded for the creator’s local device. Therefore, this project is intended **for viewing and educational purposes only**. You must adapt the paths manually if you wish to reuse or run the project on your system.

---

## 🧠 Features

- Extracts and processes text data from a zipped CSV dataset.
- Preprocesses text (cleaning, tokenization, stopword removal, lemmatization).
- Converts cleaned text into numerical form using TF-IDF vectorization.
- Trains a **Logistic Regression** classifier to detect sentiment polarity.
- Uses the **NRC Emotion Lexicon** to analyze and plot emotional expressions.
- Displays performance metrics: **accuracy**, **precision**, **recall**, and **F1-score** using a `classification_report`.
- Visualizations included:
  - Donut chart for sentiment distribution
  - Word count comparison (raw vs cleaned)
  - Emotion distribution bar chart
  - Heatmap for the confusion matrix

---

## 📊 Performance Metrics

The model evaluates sentiment predictions using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

All are shown via `classification_report` from `sklearn.metrics`.

---

## 🛠️ Tech Stack & Libraries

### 💻 Programming Language:
- Python 3.x

### 🧰 Libraries Used:

- **General-purpose**:
  - `os`, `zipfile`, `re`, `collections`
  - `pandas`, `numpy`

- **Visualization**:
  - `matplotlib.pyplot`
  - `seaborn`

- **Natural Language Processing (NLP)**:
  - `nltk`
    - `word_tokenize` – tokenizes text into words
    - `stopwords` – removes common English words
    - `WordNetLemmatizer` – reduces words to base form

- **Machine Learning**:
  - `sklearn`
    - `TfidfVectorizer` – converts text into TF-IDF vectors
    - `train_test_split` – splits data into training and testing sets
    - `LogisticRegression` – sentiment classification model
    - `classification_report`, `confusion_matrix`, `accuracy_score` – evaluation metrics

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

## 🧪 How to Use (Advanced Users)

> ⚠️ **Warning**: You will need to manually change file paths in `main_project.py` to match your system's directory structure.

## ⚠️ Portability Notice

1. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn
2.Download the NRC Emotion Lexicon if not already included.
3.Run the script after adjusting file paths accordingly.

### ✅ Recommended modification: Use `os.path.join()` and `__file__`

Instead of writing paths like this:
```python
data_path = "datasets/extracted dataset/dataset.csv"

Use this portable approach:
import os

# Get the absolute path to the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths in a system-independent way
data_path = os.path.join(base_dir, "datasets", "extracted dataset", "dataset.csv")

This ensures the project works across different operating systems and folder structures.

---

## 💡 Author Note

This project was created by **Faiz** - Undergraduate C.A. student and aspiring developer with skills in NLP, sentiment analysis, and full-cycle project implementation .
Python Developer

I personally studied and created this project, i would also encourage the people who is intrested.i have given the recommendations the run this project on your devices.Thank you !

---

