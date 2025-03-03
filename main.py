import pandas as pd
import numpy as np
import json
import re
import nltk
import time
import os
import joblib
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, 
                     kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

def load_or_process_data(file_path, force_reprocess=False):
    processed_data_path = Path('data/processed_data.pkl')
    if not force_reprocess and processed_data_path.exists():
        print("Loading processed data...")
        return joblib.load(processed_data_path)

    print("Processing data from scratch...")
    df = process_data(file_path)
    
    # Save processed data
    joblib.dump(df, processed_data_path)
    return df

def load_jsonl_in_batches(file_path, batch_size=15000):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        batch_count = 0
        for i, line in enumerate(f):
            data.append(json.loads(line))
            if (i + 1) % batch_size == 0:
                batch_count += 1
                print(f"Loaded {batch_count * batch_size} lines...")
                if batch_count >= 1:
                    break
    return data

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if len(token) > 2]
    return tokens

def save_confusion_matrix(cm, model_name, result_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    output_path = os.path.join(result_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix for {model_name} as {output_path}")

def process_data(file_path, sample_size=15000, use_sample=True):
    start_time = time.time()
    
    print("Loading data in batches...")
    reviews = load_jsonl_in_batches(file_path)
    print(f"Loaded {len(reviews)} reviews in {time.time() - start_time:.2f} seconds")

    if use_sample:
        df = pd.DataFrame(reviews[:sample_size])
        print(f"Using a sample of {sample_size} reviews")
    else:
        df = pd.DataFrame(reviews)
        print(f"Using all {len(reviews)} reviews")

    batch_size = 1000
    total_rows = len(df)
    processed_df = pd.DataFrame()

    print(f"Processing {total_rows} reviews in batches of {batch_size}...")
    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        print(f"Processing batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}...")
        
        batch = df.iloc[i:end_idx].copy()
        batch['cleaned_text'] = batch['text'].apply(clean_text)
        batch['tokens'] = batch['cleaned_text'].apply(tokenize_text)
        
        processed_df = pd.concat([processed_df, batch])

    processed_df['sentiment'] = processed_df['rating'].apply(lambda x: 1 if x >= 4.0 else 0)
    processed_df.dropna(subset=['cleaned_text', 'sentiment', 'tokens'], inplace=True)
    processed_df.drop_duplicates(subset=['cleaned_text'], inplace=True)

    return processed_df

def train_traditional_models(df, result_dir):
    X = df['cleaned_text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_vect, y_train)

    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'SVM': SVC(kernel='linear', class_weight='balanced')
    }

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test_vect)
        
        print(f"\n=== {name} ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='binary', zero_division=0):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        save_confusion_matrix(confusion_matrix(y_test, y_pred), name, result_dir)
        joblib.dump(model, os.path.join(result_dir, f'{name.lower().replace(" ", "_")}_model.joblib'))

    joblib.dump(vectorizer, os.path.join(result_dir, 'tfidf_vectorizer.joblib'))
    return X_train, X_test, y_train, y_test

def train_bert(df, result_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    train_dataset = TextDataset(X_train, y_train, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(3):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            actual_labels.extend(labels.numpy())

    print("\n=== BERT ===")
    print(f"Accuracy: {accuracy_score(actual_labels, predictions):.4f}")
    print(f"Precision: {precision_score(actual_labels, predictions, average='binary', zero_division=0):.4f}")
    print("Classification Report:")
    print(classification_report(actual_labels, predictions, zero_division=0))
    
    save_confusion_matrix(confusion_matrix(actual_labels, predictions), "BERT", result_dir)
    model.save_pretrained(os.path.join(result_dir, 'bert_model'))
    tokenizer.save_pretrained(os.path.join(result_dir, 'bert_tokenizer'))

def train_lstm(df, result_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining LSTM model on device: {device}")

    # Data preparation
    X_train, X_test, y_train, y_test = train_test_split(
        df['tokens'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    # Create vocabulary
    vocab = set()
    for tokens in df['tokens']:
        vocab.update(tokens)
    
    word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # 0 reserved for padding
    vocab_size = len(word_to_idx) + 1

    # Convert tokens to indices
    def tokens_to_indices(tokens, max_len=100):
        indices = [word_to_idx.get(word, 0) for word in tokens[:max_len]]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return indices

    X_train_indices = torch.LongTensor([tokens_to_indices(tokens) for tokens in X_train]).to(device)
    X_test_indices = torch.LongTensor([tokens_to_indices(tokens) for tokens in X_test]).to(device)
    y_train = torch.LongTensor(y_train.values).to(device)
    y_test = torch.LongTensor(y_test.values).to(device)

    # Model initialization
    model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=256,
        output_dim=2,
        n_layers=2,
        dropout=0.5
    ).to(device)

    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    n_epochs = 5
    batch_size = 64

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(X_train_indices), batch_size):
            batch_x = X_train_indices[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch: {epoch+1}, Loss: {total_loss/len(X_train_indices):.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_indices)
        predictions = test_predictions.argmax(dim=1).cpu().numpy()
        actual_labels = y_test.cpu().numpy()

    print("\n=== LSTM ===")
    print(f"Accuracy: {accuracy_score(actual_labels, predictions):.4f}")
    print(f"Precision: {precision_score(actual_labels, predictions, average='binary', zero_division=0):.4f}")
    print("Classification Report:")
    print(classification_report(actual_labels, predictions, zero_division=0))
    
    save_confusion_matrix(confusion_matrix(actual_labels, predictions), "LSTM", result_dir)
    torch.save(model.state_dict(), os.path.join(result_dir, 'lstm_model.pt'))
    joblib.dump(word_to_idx, os.path.join(result_dir, 'lstm_vocabulary.joblib'))

def train_cnn(df, result_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining CNN model on device: {device}")

    # Data preparation
    X_train, X_test, y_train, y_test = train_test_split(
        df['tokens'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    # Create vocabulary
    vocab = set()
    for tokens in df['tokens']:
        vocab.update(tokens)
    
    word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}
    vocab_size = len(word_to_idx) + 1

    # Convert tokens to indices
    def tokens_to_indices(tokens, max_len=100):
        indices = [word_to_idx.get(word, 0) for word in tokens[:max_len]]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return indices

    X_train_indices = torch.LongTensor([tokens_to_indices(tokens) for tokens in X_train]).to(device)
    X_test_indices = torch.LongTensor([tokens_to_indices(tokens) for tokens in X_test]).to(device)
    y_train = torch.LongTensor(y_train.values).to(device)
    y_test = torch.LongTensor(y_test.values).to(device)

    # Model initialization
    model = CNN(
        vocab_size=vocab_size,
        embedding_dim=100,
        n_filters=100,
        filter_sizes=[3, 4, 5],
        output_dim=2,
        dropout=0.5
    ).to(device)

    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    n_epochs = 5
    batch_size = 64

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(X_train_indices), batch_size):
            batch_x = X_train_indices[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch: {epoch+1}, Loss: {total_loss/len(X_train_indices):.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_indices)
        predictions = test_predictions.argmax(dim=1).cpu().numpy()
        actual_labels = y_test.cpu().numpy()

    print("\n=== CNN ===")
    print(f"Accuracy: {accuracy_score(actual_labels, predictions):.4f}")
    print(f"Precision: {precision_score(actual_labels, predictions, average='binary', zero_division=0):.4f}")
    print("Classification Report:")
    print(classification_report(actual_labels, predictions, zero_division=0))
    
    save_confusion_matrix(confusion_matrix(actual_labels, predictions), "CNN", result_dir)
    torch.save(model.state_dict(), os.path.join(result_dir, 'cnn_model.pt'))
    joblib.dump(word_to_idx, os.path.join(result_dir, 'cnn_vocabulary.joblib'))

def main():
    start_time = time.time()
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    
    # Load or process data
    file_path = "data/Musical_Instruments.jsonl"
    df = load_or_process_data(file_path)
    
    # Train models
    print("\nTraining traditional models...")
    train_traditional_models(df, 'result')
    
    print("\nTraining BERT model...")
    train_bert(df, 'result')
    
    print("\nTraining LSTM model...")
    train_lstm(df, 'result')
    
    print("\nTraining CNN model...")
    train_cnn(df, 'result')
    
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()

