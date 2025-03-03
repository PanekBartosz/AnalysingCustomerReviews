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
from sklearn.linear_model import LogisticRegression
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
        'Naive_Bayes': MultinomialNB(),
        'Logistic_Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'SVM': SVC(kernel='linear', class_weight='balanced', probability=True)
    }

    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        model_metrics = {}

        # Measure training time and resources
        @evaluator.measure_training_time
        def train_model():
            model.fit(X_train_resampled, y_train_resampled)
        
        _, training_metrics = train_model()
        model_metrics.update(training_metrics)

        # Cross-validation
        cv_results = evaluator.cross_validate(model, X_train_vect, y_train)
        model_metrics.update(cv_results)

        # Predictions and metrics
        y_pred = model.predict(X_test_vect)
        y_pred_proba = model.predict_proba(X_test_vect)[:, 1]

        # Calculate metrics
        model_metrics.update({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'roc_auc': evaluator.plot_roc_curve(y_test, y_pred_proba, name),
            'avg_precision': evaluator.plot_precision_recall_curve(y_test, y_pred_proba, name)
        })

        # Save confusion matrix
        save_confusion_matrix(confusion_matrix(y_test, y_pred), name, result_dir)

        # Save model metrics
        evaluator.save_metrics(name, model_metrics)
        results[name] = model_metrics

        # Save model
        joblib.dump(model, os.path.join(result_dir, f'{name.lower()}_model.joblib'))

    # Generate LaTeX comparison table
    evaluator.generate_latex_table(results)
    
    # Save vectorizer
    joblib.dump(vectorizer, os.path.join(result_dir, 'tfidf_vectorizer.joblib'))
    
    return results, vectorizer

def train_bert(df, result_dir):
    evaluator = ModelEvaluator(result_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    # Create datasets
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
    evaluator = ModelEvaluator(result_dir)
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

    return evaluation_metrics

def main():
    start_time = time.time()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('result/evaluation', exist_ok=True)
    
    # Load or process data
    file_path = "data/Musical_Instruments.jsonl"
    df = load_or_process_data(file_path)
    
    # Train and evaluate all models
    print("\nTraining and evaluating traditional models...")
    traditional_results, vectorizer = train_traditional_models(df, 'result/evaluation')
    
    print("\nTraining and evaluating BERT model...")
    bert_results = train_bert(df, 'result/evaluation')
    
    print("\nTraining and evaluating LSTM model...")
    lstm_results = train_lstm(df, 'result/evaluation')
    
    print("\nTraining and evaluating CNN model...")
    cnn_results = train_cnn(df, 'result/evaluation')
    
    # Combine all results
    all_results = {
        **traditional_results,
        'BERT': bert_results,
        'LSTM': lstm_results,
        'CNN': cnn_results
    }
    
    # Generate final comparison table
    evaluator = ModelEvaluator('result/evaluation')
    evaluator.generate_latex_table(all_results)
    
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()

