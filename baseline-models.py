import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, classification_report
from imblearn.over_sampling import RandomOverSampler

# Read data
df = pd.read_csv('data/cleaned_amazon_reviews.csv')

# Drop rows with missing values in 'cleaned_text' to avoid np.nan in vectorizer
df.dropna(subset=['cleaned_text'], inplace=True)

# Class distribution check
print("Class distribution before any resampling:")
print(df['sentiment'].value_counts())

# 4. Prepare features (X) and labels (y)
X = df['cleaned_text']
y = df['sentiment']

# Split into train and test sets 
# (stratify to keep class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000
)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Oversample the minority class in the training set
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_vect, y_train)

print("\nClass distribution AFTER oversampling (train set only):")
unique, counts = np.unique(y_train_resampled, return_counts=True)
print(dict(zip(unique, counts)))

# Naive Bayes Model
# Fit the model on the resampled data
nb_model = MultinomialNB()
nb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the original test set
y_pred_nb = nb_model.predict(X_test_vect)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average='binary', zero_division=0)

print("\n=== Naive Bayes ===")
print(f"Accuracy:  {accuracy_nb:.4f}")
print(f"Precision: {precision_nb:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_nb, zero_division=0))

# Logistic Regression Model
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
lr_model.fit(X_train_resampled, y_train_resampled)

y_pred_lr = lr_model.predict(X_test_vect)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, average='binary', zero_division=0)

print("\n=== Logistic Regression (with class_weight='balanced') ===")
print(f"Accuracy:  {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))
