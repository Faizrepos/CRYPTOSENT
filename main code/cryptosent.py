# import libraries
import os 
import zipfile
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required resources
nltk.download ('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# === STEP 1: UNZIP DATASET IF NEEDED ===
def unzip_dataset(zip_path, extract_to):
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted dataset to {extract_to}")
    else:
        print(f"✅ Dataset already extracted")

unzip_dataset(
    zip_path=r"C:\Users\camp\Downloads\archive.zip",
    extract_to=r"C:\Users\camp\Desktop\Omega.files\Omega.py"
)

# === STEP 2: LOAD DATA ===
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded with {len(df)} rows.")
        return df
    except FileNotFoundError:
        print("❌ File not found.")
        return pd.DataFrame()

df = load_data(r"C:\Users\camp\Desktop\Omega.files\Omega.py\CRYPTOSENT\datasets\Tweets.csv")

# === STEP 3: TEXT PREPROCESSING ===
def preprocess_text(text):
    sc_removed_text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(sc_removed_text.lower())     # Tokenize and lowercase
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    word_counts = {
        "original": len(tokens),
        "sc_removed": len(sc_removed_text.split()),
        "filtered": len(filtered_tokens),
        "lemmatized": len(lemmatized_tokens)
    }

    return ' '.join(lemmatized_tokens), word_counts

def preprocess_data(df):
    df[['cleaned_text', 'word_counts']] = df['text'].apply(lambda x: pd.Series(preprocess_text(x)))
    print(f"✅ Preprocessing complete on {len(df)} rows.")
    return df

df = preprocess_data(df)

# === STEP 4: TF-IDF VECTORIZATION ===
def vectorize_data(df):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text'])
    print(f"✅ TF-IDF shape: {X.shape}")
    return X, tfidf

X, tfidf = vectorize_data(df)
y = df['airline_sentiment']

# === STEP 5: TRAIN/TEST SPLIT ===
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"✅ Train/Test split: {X_train.shape[0]} train / {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(X, y)

# === STEP 6: MODEL TRAINING ===
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("✅ Model trained.")
    return model

model = train_model(X_train, y_train)

# === STEP 7: PREDICTIONS ===
def test_model(model, X_test):
    y_pred = model.predict(X_test)
    print("✅ Predictions complete.")
    return y_pred

y_pred = test_model(model, X_test)

# === STEP 8: SENTIMENT DONUT CHART ===
def visualize_sentiment_distribution(df):
    sentiment_counts = df['airline_sentiment'].value_counts()
    labels = sentiment_counts.index
    sizes = sentiment_counts.values

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=['#E63946', '#8D99AE', '#A8DADC'],
        autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4)
    )
    plt.setp(autotexts, size=12, weight="bold")
    ax.set_title("Sentiment Distribution", fontsize=14, fontweight='bold')
    
visualize_sentiment_distribution(df)

# === STEP 9: WORD COUNT BAR CHART ===
def visualize_word_counts(df):
    word_counts = df['word_counts'].apply(pd.Series).sum()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Original", "spl-char removed", "Filtered", "Lemmatized"], word_counts, color=['#1B4F72', '#2874A6', '#73C6B6', '#A2D9CE'])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                str(int(bar.get_height())), ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
    ax.set_title("Text Preprocessing Overview", fontsize=14, fontweight='bold')
    ax.set_ylabel("Word Count", fontsize=14, fontweight='bold')
    plt.tight_layout()
    

visualize_word_counts(df)

# === STEP 10: EMOTION DETECTION USING NRC LEXICON ===
def load_emotion_lexicon(path):
    lexicon_df = pd.read_csv(path, sep='\t', names=['word', 'emotion', 'association'])
    lexicon_df = lexicon_df[lexicon_df['association'] == 1]
    emotion_lexicon = defaultdict(list)
    for _, row in lexicon_df.iterrows():
        emotion_lexicon[row['word']].append(row['emotion'])
    print("✅ Emotion lexicon loaded.")
    return emotion_lexicon

def get_emotion_counts(texts, lexicon):
    emotion_counts = Counter()
    for text in texts:
        for word in text.split():
            emotion_counts.update(lexicon.get(word, []))
    return dict(emotion_counts)

def visualize_emotions(emotion_counts):
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    plt.figure(figsize=(10, 6))
    plt.bar(emotions, counts, color=plt.cm.PuBuGn(np.linspace(0.3, 0.8, len(emotions))))
    plt.title('Emotion Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Emotions', fontsize=14, fontweight='bold')
    plt.ylabel('Counts', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.tight_layout()
    

emotion_lexicon = load_emotion_lexicon(
    r"C:\Users\camp\Desktop\Omega.files\Omega.py\CRYPTOSENT\essential libs, dicts\NRC-Emotion-Lexicon\NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
)
emotion_counts = get_emotion_counts(df['cleaned_text'], emotion_lexicon)
visualize_emotions(emotion_counts)

# === STEP 11: CONFUSION MATRIX HEATMAP ===
def plot_confusion_matrix(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix Heatmap", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted", fontsize=14, fontweight='bold')
    plt.ylabel("True", fontsize=14, fontweight='bold')
    plt.tight_layout()
    

plot_confusion_matrix(y_test, y_pred)

# === STEP 12: REPORT METRICS ===
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
plt.show()