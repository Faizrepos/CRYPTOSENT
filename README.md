# CRYPTOSENT: Sentiment & Emotion Analysis Tool

CRYPTOSENT is a custom-built NLP pipeline for sentiment and emotion analysis, designed with modular Python scripts and visualizations. It covers everything from data preprocessing to model training, prediction, and chart-based analytics — including sarcasm and irony detection.

> 🚨 **NOTE:** This project was created for **demonstration and viewing purposes only**. The paths used in the main script (`main_project.py`) are designed specifically for the creator's local development environment. Attempting to run this code on another device without modifying the paths will likely result in errors.

---

## 🔍 Features

- Automatic dataset extraction from `.zip` file
- Custom text preprocessing pipeline
- TF-IDF vectorization
- Sentiment classification using logistic regression model
- Emotion detection via NRC Emotion Lexicon
- Multiple data visualizations:
  - Donut pie chart for sentiment distribution
  - Bar chart comparison (word count: before vs after preprocessing)
  - Emotion frequency bar chart
  - Confusion matrix heatmap
- Modular function-based structure
- Sarcasm & irony handling (basic detection)

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
