import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the data from the JSON file
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Convert the JSON data to a DataFrame
df = pd.DataFrame(data)

# Preprocess the text data (example of lowercasing and removing punctuation)
df['text'] = df['text'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Flatten the error labels for simplicity
# Assuming we are interested in the last error label for each example
df['error_label'] = df['error_labels'].apply(lambda x: x[-1] if isinstance(x[-1], str) else 'No_Error')

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['error_label'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Random Forest classifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model_rf.predict(X_test_tfidf)

# Evaluate the model
all_labels = list(label_encoder.classes_)
print("Random Forest Classifier Results:")
print(classification_report(y_test, y_pred, target_names=all_labels, labels=list(range(len(all_labels)))))
print(confusion_matrix(y_test, y_pred, labels=list(range(len(all_labels)))))

# Tokenization and padding for neural network
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# Build the neural network model
model_nn = Sequential()
model_nn.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model_nn.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model_nn.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the neural network model
model_nn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the neural network model
model_nn.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))

# Evaluate the neural network model
loss, accuracy = model_nn.evaluate(X_test_pad, y_test)
print(f'Neural Network Model Results: Loss: {loss}, Accuracy: {accuracy}')

# Test cases
test_cases = [
    {"text": "Human 1: Hi! মানব 1: হাই!", "expected_label": "No_Error"},
    {"text": "Human 1: Hello! মানব 1: বিদায়!", "expected_label": "Mistranslation-Neutral"},
    {"text": "Human 1: How are you? মানব 1: তুমি কেমন আছো?", "expected_label": "Spelling-Neutral"},
    {"text": "Human 1: What is your name? মানব 1: তোমার নাম কি?", "expected_label": "Omission-Critical"},
    {"text": "Human 1: I'm fine, thank you. মানব 1: আমি ভালো আছি, ধন্যবাদ.", "expected_label": "Addition-Major"},
    {"text": "Human 1: I go to school. মানব 1: আমি যাই স্কুলে.", "expected_label": "Fluency-Major"}
]

# Evaluate test cases
for test in test_cases:
    text = test['text'].lower()
    text = re.sub(r'[^\w\s]', '', text)
    X_test_tfidf = vectorizer.transform([text])
    predicted_label_index = model_rf.predict(X_test_tfidf)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    
    print(f"Input: {test['text']}")
    print(f"Expected Label: {test['expected_label']}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Result: {'Pass' if predicted_label == test['expected_label'] else 'Fail'}")
    print("-" * 50)
