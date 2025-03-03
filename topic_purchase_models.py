import pandas as pd
import numpy as np
import json
import os
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import psutil
import time

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import (accuracy_score, precision_score, mean_squared_error,
                           mean_absolute_error, r2_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler

class ModelEvaluator:
    def __init__(self, result_dir='result/topic_purchase_evaluation'):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def measure_training_time(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_cpu = psutil.Process().cpu_percent()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            result = func(*args, **kwargs)

            end_time = time.time()
            end_cpu = psutil.Process().cpu_percent()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            metrics = {
                'training_time': end_time - start_time,
                'cpu_usage': end_cpu - start_cpu,
                'memory_usage': end_memory - start_memory
            }

            self.metrics[func.__name__] = metrics
            return result, metrics
        return wrapper

    def save_metrics(self, model_name, metrics):
        metrics_file = self.result_dir / f'metrics_{model_name}_{self.timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

def save_confusion_matrix(cm, model_name, result_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    output_path = os.path.join(result_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.savefig(output_path)
    plt.close()

def train_topic_models(df, result_dir):
    """
    Train and evaluate topic identification models (LDA and NMF)
    """
    print("\nTraining topic identification models...")
    evaluator = ModelEvaluator(result_dir)
    
    # Data preparation
    print("Preparing data for topic modeling...")
    count_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    doc_term_matrix = count_vectorizer.fit_transform(df['cleaned_text'])
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    
    # Number of topics to identify
    n_topics = 5
    
    # LDA
    print("\nTraining LDA model...")
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        n_jobs=-1
    )
    
    lda_output = lda_model.fit_transform(doc_term_matrix)
    
    # NMF
    print("\nTraining NMF model...")
    nmf_model = NMF(
        n_components=n_topics,
        random_state=42
    )
    
    nmf_output = nmf_model.fit_transform(tfidf_matrix)
    
    # Evaluate topic coherence
    def calculate_topic_coherence(model, feature_names, doc_term_matrix, top_n=10):
        topic_coherence = []
        for topic_idx in range(model.n_components):
            top_words_idx = model.components_[topic_idx].argsort()[:-top_n-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_coherence.append(len(set(top_words)) / top_n)  # Simple coherence metric
        return np.mean(topic_coherence)
    
    # Get feature names
    count_feature_names = count_vectorizer.get_feature_names_out()
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Calculate metrics
    lda_coherence = calculate_topic_coherence(lda_model, count_feature_names, doc_term_matrix)
    nmf_coherence = calculate_topic_coherence(nmf_model, tfidf_feature_names, tfidf_matrix)
    
    # Save top words for each topic
    def save_top_words(model, feature_names, model_name, n_words=10):
        topics = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_words-1:-1]
            topics[f'Topic {topic_idx + 1}'] = [feature_names[i] for i in top_words_idx]
        
        with open(os.path.join(result_dir, f'{model_name}_topics.json'), 'w') as f:
            json.dump(topics, f, indent=4)
    
    save_top_words(lda_model, count_feature_names, 'LDA')
    save_top_words(nmf_model, tfidf_feature_names, 'NMF')
    
    # Save models
    joblib.dump(lda_model, os.path.join(result_dir, 'lda_model.joblib'))
    joblib.dump(nmf_model, os.path.join(result_dir, 'nmf_model.joblib'))
    
    # Save vectorizers
    joblib.dump(count_vectorizer, os.path.join(result_dir, 'count_vectorizer.joblib'))
    joblib.dump(tfidf_vectorizer, os.path.join(result_dir, 'tfidf_vectorizer.joblib'))
    
    # Prepare and save metrics
    lda_metrics = {
        'coherence': lda_coherence,
        'perplexity': lda_model.perplexity(doc_term_matrix),
        'n_topics': n_topics
    }
    
    nmf_metrics = {
        'coherence': nmf_coherence,
        'reconstruction_error': nmf_model.reconstruction_err_,
        'n_topics': n_topics
    }
    
    evaluator.save_metrics('LDA', lda_metrics)
    evaluator.save_metrics('NMF', nmf_metrics)
    
    return {
        'LDA': lda_metrics,
        'NMF': nmf_metrics
    }

def train_purchase_models(df, result_dir):
    """
    Train and evaluate purchase behavior prediction models
    """
    print("\nTraining purchase behavior prediction models...")
    evaluator = ModelEvaluator(result_dir)
    
    # Data preparation
    print("Preparing data for purchase prediction...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_text'])
    
    # Prepare dependent variables
    y_binary = (df['rating'] >= 4.0).astype(int)  # Purchase/no purchase classification
    y_regression = df['rating']  # Rating prediction as a proxy for spending
    
    # Data split
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    # Logistic Regression
    print("\nTraining Logistic Regression model...")
    log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
    log_reg.fit(X_train_clf, y_train_clf)
    
    log_reg_pred = log_reg.predict(X_test_clf)
    log_reg_metrics = {
        'accuracy': accuracy_score(y_test_clf, log_reg_pred),
        'precision': precision_score(y_test_clf, log_reg_pred)
    }
    
    # Random Forest
    print("\nTraining Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X_train_clf, y_train_clf)
    
    rf_pred = rf.predict(X_test_clf)
    rf_metrics = {
        'accuracy': accuracy_score(y_test_clf, rf_pred),
        'precision': precision_score(y_test_clf, rf_pred)
    }
    
    # Linear Regression
    print("\nTraining Linear Regression model...")
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train_reg, y_train_reg)
    
    lr_pred = lr.predict(X_test_reg)
    lr_metrics = {
        'mse': mean_squared_error(y_test_reg, lr_pred),
        'mae': mean_absolute_error(y_test_reg, lr_pred),
        'r2': r2_score(y_test_reg, lr_pred)
    }
    
    # SVR
    print("\nTraining SVR model...")
    svr = SVR(kernel='rbf')
    svr.fit(X_train_reg, y_train_reg)
    
    svr_pred = svr.predict(X_test_reg)
    svr_metrics = {
        'mse': mean_squared_error(y_test_reg, svr_pred),
        'mae': mean_absolute_error(y_test_reg, svr_pred),
        'r2': r2_score(y_test_reg, svr_pred)
    }
    
    # Save confusion matrices
    save_confusion_matrix(
        confusion_matrix(y_test_clf, log_reg_pred),
        "Logistic_Regression",
        result_dir
    )
    save_confusion_matrix(
        confusion_matrix(y_test_clf, rf_pred),
        "Random_Forest",
        result_dir
    )
    
    # Save models
    joblib.dump(log_reg, os.path.join(result_dir, 'logistic_regression_model.joblib'))
    joblib.dump(rf, os.path.join(result_dir, 'random_forest_model.joblib'))
    joblib.dump(lr, os.path.join(result_dir, 'linear_regression_model.joblib'))
    joblib.dump(svr, os.path.join(result_dir, 'svr_model.joblib'))
    joblib.dump(vectorizer, os.path.join(result_dir, 'purchase_vectorizer.joblib'))
    
    # Save metrics
    evaluator.save_metrics('Logistic_Regression', log_reg_metrics)
    evaluator.save_metrics('Random_Forest', rf_metrics)
    evaluator.save_metrics('Linear_Regression', lr_metrics)
    evaluator.save_metrics('SVR', svr_metrics)
    
    return {
        'Logistic_Regression': log_reg_metrics,
        'Random_Forest': rf_metrics,
        'Linear_Regression': lr_metrics,
        'SVR': svr_metrics
    }

def collect_and_compare_results(result_dir, model_type='topic'):
    """
    Collect and compare results from all models
    """
    print(f"\nCollecting results for {model_type} models...")
    
    if model_type == 'topic':
        model_names = ['LDA', 'NMF']
        metrics_of_interest = ['coherence', 'perplexity', 'reconstruction_error']
    else:  # purchase
        model_names = ['Logistic_Regression', 'Random_Forest', 'Linear_Regression', 'SVR']
        metrics_of_interest = ['accuracy', 'precision', 'mse', 'mae', 'r2']
    
    all_results = {}
    
    for model_name in model_names:
        metrics_file = Path(result_dir) / f'metrics_{model_name}_*.json'
        try:
            files = list(Path(result_dir).glob(f'metrics_{model_name}_*.json'))
            if files:
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    metrics = json.load(f)
                
                model_metrics = {}
                for metric in metrics_of_interest:
                    if metric in metrics:
                        model_metrics[metric] = metrics[metric]
                
                all_results[model_name] = model_metrics
                
        except Exception as e:
            print(f"Could not load results for {model_name}: {str(e)}")
    
    # Create comparison table
    metrics_df = pd.DataFrame.from_dict(all_results, orient='columns')
    metrics_df = metrics_df.round(6)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metrics_df.to_csv(Path(result_dir) / f'{model_type}_comparison_{timestamp}.csv')
    metrics_latex = metrics_df.to_latex()
    
    with open(Path(result_dir) / f'{model_type}_comparison_{timestamp}.tex', 'w') as f:
        f.write(metrics_latex)
    
    print(f"\n{model_type.capitalize()} Model Comparison:")
    print(metrics_df)
    
    return all_results

def main():
    start_time = time.time()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('result/topic_purchase_evaluation', exist_ok=True)
    
    # Load data
    df = joblib.load('data/processed_data.pkl')
    
    # Train and evaluate topic models
    print("\nTraining and evaluating topic identification models...")
    topic_results = train_topic_models(df, 'result/topic_purchase_evaluation')
    topic_comparison = collect_and_compare_results('result/topic_purchase_evaluation', 'topic')
    
    # Train and evaluate purchase prediction models
    print("\nTraining and evaluating purchase prediction models...")
    purchase_results = train_purchase_models(df, 'result/topic_purchase_evaluation')
    purchase_comparison = collect_and_compare_results('result/topic_purchase_evaluation', 'purchase')
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 