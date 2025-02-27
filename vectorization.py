import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import time

start_time = time.time()

# Load the cleaned data
try:
    print("Loading cleaned Amazon reviews data...")
    df = pd.read_csv('cleaned_amazon_reviews.csv')
    print(f"Loaded {len(df)} reviews")
except FileNotFoundError:
    print("Error: Could not find cleaned_amazon_reviews.csv")
    print("Make sure you've run the previous data cleaning script first")
    exit(1)

# Check if we have the needed columns
if 'cleaned_text' not in df.columns:
    print("Error: 'cleaned_text' column not found in the data")
    exit(1)

# For tokens - convert from string representation to actual list if needed
if 'tokens' in df.columns and isinstance(df['tokens'].iloc[0], str):
    print("Converting tokens from string to list...")
    df['tokens'] = df['tokens'].apply(lambda x: x.strip('[]').replace("'", "").split(', '))

# For reviews with a target variable
target_column = 'rating'

# Drop rows with missing target
df = df.dropna(subset=[target_column])

# Check that we have data to work with
if len(df) == 0:
    print("Error: No data available after preprocessing")
    exit(1)

# Drop rows where 'cleaned_text' is NaN or empty
df = df.dropna(subset=['cleaned_text'])
df = df[df['cleaned_text'].str.strip() != '']  # Remove empty strings

# Reset index after dropping
df = df.reset_index(drop=True)

# Vectorization - TF-IDF
print("Performing TF-IDF Vectorization...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit features to prevent memory issues
    min_df=5,          # Minimum document frequency
    max_df=0.8,        # Maximum document frequency
    stop_words='english'
)

# Fit and transform the cleaned text
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])
print(f"TF-IDF vectorization complete: {X_tfidf.shape[0]} documents, {X_tfidf.shape[1]} features")

# Vectorization - Bag of Words/Count Vectorizer
print("Performing Count Vectorization...")
count_vectorizer = CountVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8,
    stop_words='english'
)

X_bow = count_vectorizer.fit_transform(df['cleaned_text'])
print(f"Count vectorization complete: {X_bow.shape[0]} documents, {X_bow.shape[1]} features")

# Train-Test Split
test_size = 0.2  # 20% for testing
random_state = 42  # For reproducibility

y = df[target_column]

# Split the TF-IDF features
X_tfidf_train, X_tfidf_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=test_size, random_state=random_state
)

# Split the Bag of Words features
X_bow_train, X_bow_test, _, _ = train_test_split(
    X_bow, y, test_size=test_size, random_state=random_state
)

print(f"Data split into train ({len(y_train)} samples) and test ({len(y_test)} samples) sets")

# Save the vectorizers and the split data
print("Saving vectorizers and datasets...")

# Save vectorizers
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(count_vectorizer, f)

# Save datasets
np.savez('tfidf_train_test_data.npz', 
            X_train=X_tfidf_train, X_test=X_tfidf_test, 
            y_train=y_train, y_test=y_test)

np.savez('bow_train_test_data.npz', 
            X_train=X_bow_train, X_test=X_bow_test, 
            y_train=y_train, y_test=y_test)

# Save dataset information for reference
dataset_info = {
    'original_size': len(df),
    'feature_count_tfidf': X_tfidf.shape[1],
    'feature_count_bow': X_bow.shape[1],
    'train_size': X_tfidf_train.shape[0],
    'test_size': X_tfidf_test.shape[0],
    'has_target': target_column is not None,
    'target_column': target_column
}

# Save as JSON
import json
with open('dataset_info.json', 'w') as f:
    json.dump(dataset_info, f, indent=4)

execution_time = time.time() - start_time
print(f"\nVectorization and train/test split completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
print(f"Files saved: tfidf_vectorizer.pkl, count_vectorizer.pkl, tfidf_train_test_data.npz, bow_train_test_data.npz, dataset_info.json")