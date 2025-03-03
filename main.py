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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, classification_report, 
                           confusion_matrix, roc_curve, auc, precision_recall_curve, 
                           average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
from datetime import datetime
import psutil
from sklearn.decomposition import TruncatedSVD

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Konwersja do list dla bezpieczniejszego indeksowania
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx >= len(self.texts):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.texts)}")
            
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
        
        # Inicjalizacja wag
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)  # [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)  # [batch size, 1, sent len, emb dim]
        
        # Zastosuj konwolucje i aktywację
        conved = []
        for conv in self.convs:
            # conv(embedded) = [batch size, n_filters, sent len - filter_size + 1, 1]
            conv_out = conv(embedded)
            conv_out = conv_out.squeeze(3)  # [batch size, n_filters, sent len - filter_size + 1]
            conv_out = nn.functional.relu(conv_out)
            conved.append(conv_out)
            
        # Pooling
        pooled = []
        for conv_out in conved:
            # max_pool = [batch size, n_filters, 1]
            pooled_out = nn.functional.max_pool1d(conv_out, conv_out.shape[2])
            pooled_out = pooled_out.squeeze(2)  # [batch size, n_filters]
            pooled.append(pooled_out)
            
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class ModelEvaluator:
    def __init__(self, result_dir='result/evaluation'):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def measure_training_time(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_cpu = psutil.Process().cpu_percent()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            result = func(*args, **kwargs)

            end_time = time.time()
            end_cpu = psutil.Process().cpu_percent()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            metrics = {
                'training_time': end_time - start_time,
                'cpu_usage': end_cpu - start_cpu,
                'memory_usage': end_memory - start_memory
            }

            self.metrics[func.__name__] = metrics
            return result, metrics
        return wrapper

    def cross_validate(self, model, X, y, cv=5):
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }

    def plot_roc_curve(self, y_true, y_pred_proba, model_name):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        plt.savefig(self.result_dir / f'roc_curve_{model_name}_{self.timestamp}.png')
        plt.close()
        
        return roc_auc

    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        
        plt.savefig(self.result_dir / f'pr_curve_{model_name}_{self.timestamp}.png')
        plt.close()
        
        return avg_precision

    def save_metrics(self, model_name, metrics):
        metrics_file = self.result_dir / f'metrics_{model_name}_{self.timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

    def generate_latex_table(self, metrics_dict):
        df = pd.DataFrame(metrics_dict).round(4)
        latex_table = df.to_latex()
        
        with open(self.result_dir / f'comparison_table_{self.timestamp}.tex', 'w') as f:
            f.write(latex_table)
        
        return latex_table

def plot_training_curves(metrics, model_name, result_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(metrics['epoch_losses'])
    ax1.set_title(f'{model_name} Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    # Memory usage
    ax2.plot(metrics['memory_usage'], label='RAM')
    if metrics.get('gpu_memory_usage'):
        ax2.plot(metrics['gpu_memory_usage'], label='GPU')
    ax2.set_title(f'{model_name} Memory Usage')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Memory (MB)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(Path(result_dir) / f'training_curves_{model_name}.png')
    plt.close()

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
    evaluator = ModelEvaluator(result_dir)
    X = df['cleaned_text']
    y = df['sentiment']

    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Wektoryzacja
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Oversampling
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_vect, y_train)

    # Tylko model SVM
    model = SGDClassifier(
        loss='hinge',  # Linear SVM
        class_weight='balanced',
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    )

    # Trenowanie
    print("\nTrenowanie modelu SVM...")
    model.fit(X_train_resampled, y_train_resampled)

    # Predykcje
    y_pred = model.predict(X_test_vect)

    # Obliczanie metryk
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nWyniki klasyfikacji sentymentu:")
    print(f"Dokładność (accuracy): {accuracy:.4f}")
    print(f"Precyzja (precision): {precision:.4f}")
    print("\nMacierz pomyłek:")
    print(conf_matrix)

    # Zapisywanie macierzy pomyłek
    save_confusion_matrix(conf_matrix, "SVM", result_dir)

    # Zapisywanie modelu
    joblib.dump(model, os.path.join(result_dir, 'svm_model.joblib'))
    joblib.dump(vectorizer, os.path.join(result_dir, 'tfidf_vectorizer.joblib'))

    return {
        'model': model,
        'vectorizer': vectorizer,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'confusion_matrix': conf_matrix
        }
    }

def train_bert(df, result_dir):
    evaluator = ModelEvaluator(result_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    # Przygotowanie danych - dodaj reset_index()
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['sentiment']
    )
    
    # Reset indeksów po podziale
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Tworzenie datasetów
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Training setup
    optimizer = AdamW(model.parameters(), lr=2e-5)
    training_metrics = {
        'epoch_times': [],
        'epoch_losses': [],
        'memory_usage': [],
        'gpu_memory_usage': [] if torch.cuda.is_available() else None
    }

    # Training loop
    n_epochs = 3
    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Record metrics
        epoch_time = time.time() - epoch_start
        training_metrics['epoch_times'].append(epoch_time)
        training_metrics['epoch_losses'].append(total_loss / len(train_loader))
        training_metrics['memory_usage'].append(
            psutil.Process().memory_info().rss / 1024 / 1024
        )
        
        if torch.cuda.is_available():
            training_metrics['gpu_memory_usage'].append(
                torch.cuda.memory_allocated() / 1024 / 1024
            )

    # Evaluation
    model.eval()
    predictions = []
    actual_labels = []
    proba_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            proba = torch.softmax(outputs.logits, dim=1)
            predictions.extend(proba.argmax(dim=1).cpu().numpy())
            proba_predictions.extend(proba[:, 1].cpu().numpy())
            actual_labels.extend(labels.numpy())

    # Calculate metrics
    evaluation_metrics = {
        'accuracy': accuracy_score(actual_labels, predictions),
        'precision': precision_score(actual_labels, predictions),
        'roc_auc': evaluator.plot_roc_curve(actual_labels, proba_predictions, "BERT"),
        'avg_precision': evaluator.plot_precision_recall_curve(actual_labels, proba_predictions, "BERT"),
        'training_metrics': training_metrics
    }

    # Save confusion matrix
    save_confusion_matrix(confusion_matrix(actual_labels, predictions), "BERT", result_dir)

    # Save metrics and model
    evaluator.save_metrics("BERT", evaluation_metrics)
    model.save_pretrained(os.path.join(result_dir, 'bert_model'))
    tokenizer.save_pretrained(os.path.join(result_dir, 'bert_tokenizer'))

    # Plot training curves
    plot_training_curves(training_metrics, "BERT", result_dir)

    return evaluation_metrics

def train_lstm(df, result_dir):
    evaluator = ModelEvaluator(result_dir)
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
    model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=256,
        output_dim=2,
        n_layers=2,
        dropout=0.5
    ).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    training_metrics = {
        'epoch_times': [],
        'epoch_losses': [],
        'memory_usage': [],
        'gpu_memory_usage': [] if torch.cuda.is_available() else None
    }

    # Training loop
    n_epochs = 5
    batch_size = 64

    for epoch in range(n_epochs):
        epoch_start = time.time()
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

        # Record metrics
        epoch_time = time.time() - epoch_start
        training_metrics['epoch_times'].append(epoch_time)
        training_metrics['epoch_losses'].append(total_loss/len(X_train_indices))
        training_metrics['memory_usage'].append(
            psutil.Process().memory_info().rss / 1024 / 1024
        )
        
        if torch.cuda.is_available():
            training_metrics['gpu_memory_usage'].append(
                torch.cuda.memory_allocated() / 1024 / 1024
            )
        
        print(f'Epoch: {epoch+1}, Loss: {total_loss/len(X_train_indices):.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_indices)
        proba = torch.softmax(test_predictions, dim=1)
        predictions = proba.argmax(dim=1).cpu().numpy()
        proba_predictions = proba[:, 1].cpu().numpy()
        actual_labels = y_test.cpu().numpy()

    # Calculate metrics
    evaluation_metrics = {
        'accuracy': accuracy_score(actual_labels, predictions),
        'precision': precision_score(actual_labels, predictions),
        'roc_auc': evaluator.plot_roc_curve(actual_labels, proba_predictions, "LSTM"),
        'avg_precision': evaluator.plot_precision_recall_curve(actual_labels, proba_predictions, "LSTM"),
        'training_metrics': training_metrics
    }

    # Save results
    save_confusion_matrix(confusion_matrix(actual_labels, predictions), "LSTM", result_dir)
    evaluator.save_metrics("LSTM", evaluation_metrics)
    torch.save(model.state_dict(), os.path.join(result_dir, 'lstm_model.pt'))
    joblib.dump(word_to_idx, os.path.join(result_dir, 'lstm_vocabulary.joblib'))

    # Plot training curves
    plot_training_curves(training_metrics, "LSTM", result_dir)

    return evaluation_metrics

def train_cnn(df, result_dir):
    try:
        evaluator = ModelEvaluator(result_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nTraining CNN model on device: {device}")
        
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

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

        def tokens_to_indices(tokens, max_len=100):
            indices = [word_to_idx.get(word, 0) for word in tokens[:max_len]]
            if len(indices) < max_len:
                indices += [0] * (max_len - len(indices))
            return indices

        print("Converting tokens to indices...")
        X_train_indices = torch.LongTensor([tokens_to_indices(tokens) for tokens in X_train]).to(device)
        X_test_indices = torch.LongTensor([tokens_to_indices(tokens) for tokens in X_test]).to(device)
        y_train = torch.LongTensor(y_train.values).to(device)
        y_test = torch.LongTensor(y_test.values).to(device)

        # Model initialization
        model = CNN(
            vocab_size=vocab_size,
            embedding_dim=100,
            n_filters=100,
            filter_sizes=[2, 3, 4],  # Reduced filter sizes
            output_dim=2,
            dropout=0.3  # Reduced dropout
        ).to(device)

        # Training configuration
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
        training_metrics = {
            'epoch_times': [],
            'epoch_losses': [],
            'memory_usage': [],
            'gpu_memory_usage': [] if torch.cuda.is_available() else None
        }

        # Training parameters
        n_epochs = 5
        batch_size = 32  # Reduced batch size
        grad_clip = 1.0  # Gradient clipping

        print("\nStarting training...")
        for epoch in range(n_epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            batch_count = 0
            
            # Shuffle data
            indices = torch.randperm(len(X_train_indices))
            X_train_indices = X_train_indices[indices]
            y_train = y_train[indices]
            
            for i in range(0, len(X_train_indices), batch_size):
                batch_x = X_train_indices[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                
                try:
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    if batch_count % 50 == 0:
                        print(f'Epoch: {epoch+1}, Batch: {batch_count}, Loss: {loss.item():.4f}')
                        
                except RuntimeError as e:
                    print(f"Error during batch training: {e}")
                    continue

            # Save metrics
            avg_loss = total_loss / batch_count
            epoch_time = time.time() - epoch_start
            training_metrics['epoch_times'].append(epoch_time)
            training_metrics['epoch_losses'].append(avg_loss)
            training_metrics['memory_usage'].append(
                psutil.Process().memory_info().rss / 1024 / 1024
            )
            
            if torch.cuda.is_available():
                training_metrics['gpu_memory_usage'].append(
                    torch.cuda.memory_allocated() / 1024 / 1024
                )
            
            print(f'Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')
            
            # Clear CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluation
        print("\nStarting evaluation...")
        model.eval()
        predictions = []
        actual_labels = []
        proba_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_test_indices), batch_size):
                batch_x = X_test_indices[i:i+batch_size]
                batch_y = y_test[i:i+batch_size]
                
                outputs = model(batch_x)
                proba = torch.softmax(outputs, dim=1)
                predictions.extend(proba.argmax(dim=1).cpu().numpy())
                proba_predictions.extend(proba[:, 1].cpu().numpy())
                actual_labels.extend(batch_y.cpu().numpy())

        # Calculate metrics
        evaluation_metrics = {
            'accuracy': accuracy_score(actual_labels, predictions),
            'precision': precision_score(actual_labels, predictions),
            'roc_auc': evaluator.plot_roc_curve(actual_labels, proba_predictions, "CNN"),
            'avg_precision': evaluator.plot_precision_recall_curve(actual_labels, proba_predictions, "CNN"),
            'training_metrics': training_metrics
        }

        # Save results
        save_confusion_matrix(confusion_matrix(actual_labels, predictions), "CNN", result_dir)
        evaluator.save_metrics("CNN", evaluation_metrics)
        torch.save(model.state_dict(), os.path.join(result_dir, 'cnn_model.pt'))
        joblib.dump(word_to_idx, os.path.join(result_dir, 'cnn_vocabulary.joblib'))

        # Plot training curves
        plot_training_curves(training_metrics, "CNN", result_dir)

        print("\nCNN Model Results:")
        print(f"Accuracy: {evaluation_metrics['accuracy']:.4f}")
        print(f"Precision: {evaluation_metrics['precision']:.4f}")
        print(f"ROC AUC: {evaluation_metrics['roc_auc']:.4f}")
        
        return evaluation_metrics
        
    except Exception as e:
        print(f"\nAn error occurred during CNN model training: {str(e)}")
        print("Error details:")
        import traceback
        traceback.print_exc()
        return None

def collect_and_compare_results(result_dir):
    """
    Collect results from all trained models and generate comparison tables and plots.
    """
    print("\nCollecting results from all models...")
    
    # Initialize results dictionary
    all_results = {}
    
    # Load results for each model
    model_names = ['SVM', 'Naive_Bayes', 'Logistic_Regression', 'BERT', 'LSTM', 'CNN']
    
    for model_name in model_names:
        metrics_file = Path(result_dir) / f'metrics_{model_name}_*.json'
        try:
            # Get the most recent metrics file for each model
            files = list(Path(result_dir).glob(f'metrics_{model_name}_*.json'))
            if files:
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    metrics = json.load(f)
                
                # Extract all available metrics
                model_metrics = {}
                
                # Training time metrics
                if 'training_metrics' in metrics:
                    training_metrics = metrics['training_metrics']
                    model_metrics['training_time'] = sum(training_metrics.get('epoch_times', [0]))
                    model_metrics['cpu_usage'] = training_metrics.get('cpu_usage', 0)
                    model_metrics['memory_usage'] = max(training_metrics.get('memory_usage', [0]))
                
                # Cross-validation metrics
                if 'mean_accuracy' in metrics:
                    model_metrics['mean_accuracy'] = metrics['mean_accuracy']
                    model_metrics['std_accuracy'] = metrics['std_accuracy']
                    model_metrics['cv_scores'] = metrics['cv_scores']
                
                # Performance metrics
                model_metrics['accuracy'] = metrics.get('accuracy', None)
                model_metrics['precision'] = metrics.get('precision', None)
                model_metrics['roc_auc'] = metrics.get('roc_auc', None)
                model_metrics['avg_precision'] = metrics.get('avg_precision', None)
                
                all_results[model_name] = model_metrics
                
        except Exception as e:
            print(f"Could not load results for {model_name}: {str(e)}")
    
    # Create comparison table
    metrics_df = pd.DataFrame.from_dict(all_results, orient='columns')
    metrics_df = metrics_df.round(6)
    
    # Save tables
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    metrics_df.to_csv(Path(result_dir) / f'model_comparison_full_{timestamp}.csv')
    
    # Generate LaTeX table
    metrics_latex = metrics_df.to_latex()
    with open(Path(result_dir) / f'model_comparison_full_{timestamp}.tex', 'w') as f:
        f.write(metrics_latex)
    
    # Print results
    print("\nFull Model Comparison:")
    print(metrics_df)
    
    # Create comparison plots
    plt.figure(figsize=(15, 5))
    
    # Accuracy comparison
    plt.subplot(1, 3, 1)
    accuracies = [all_results[model].get('accuracy', 0) for model in model_names if model in all_results]
    plt.bar(range(len(accuracies)), accuracies)
    plt.xticks(range(len(accuracies)), [model for model in model_names if model in all_results], rotation=45)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    
    # ROC AUC comparison
    plt.subplot(1, 3, 2)
    roc_aucs = [all_results[model].get('roc_auc', 0) for model in model_names if model in all_results]
    plt.bar(range(len(roc_aucs)), roc_aucs)
    plt.xticks(range(len(roc_aucs)), [model for model in model_names if model in all_results], rotation=45)
    plt.title('ROC AUC Comparison')
    plt.ylabel('ROC AUC')
    
    # Training time comparison
    plt.subplot(1, 3, 3)
    times = [all_results[model].get('training_time', 0) for model in model_names if model in all_results]
    plt.bar(range(len(times)), times)
    plt.xticks(range(len(times)), [model for model in model_names if model in all_results], rotation=45)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(Path(result_dir) / f'model_comparison_plots_{timestamp}.png')
    plt.close()
    
    return all_results

def main():
    start_time = time.time()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('result/evaluation', exist_ok=True)
    
    # Generate comparison of all models
    print("\nGenerating comparison of all models...")
    comparison_results = collect_and_compare_results('result/evaluation')
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()

