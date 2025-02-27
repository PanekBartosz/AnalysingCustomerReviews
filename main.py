import pandas as pd
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time

start_time = time.time()

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Path to the data
file_path = "Musical_Instruments.jsonl"

# Function to load the jsonl file 
def load_jsonl_in_batches(file_path, batch_size=10000):
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

print("Loading data in batches...")
reviews = load_jsonl_in_batches(file_path)
print(f"Loaded {len(reviews)} reviews in {time.time() - start_time:.2f} seconds")

# Sample size
sample_size = 10000
use_sample = True

# Convert to DataFrame
if use_sample:
    df = pd.DataFrame(reviews[:sample_size])
    print(f"Using a sample of {sample_size} reviews")
else:
    df = pd.DataFrame(reviews)
    print(f"Using all {len(reviews)} reviews")

# Data cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Tokenization function
def tokenize_text(text):

    text = clean_text(text)

    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove tokens that are too short
    tokens = [token for token in tokens if len(token) > 2]
    
    return tokens

# Column containing the review text
review_column = 'text'

# Process in batches
batch_size = 1000
total_rows = len(df)
processed_df = pd.DataFrame()

print(f"Processing {total_rows} reviews in batches of {batch_size}...")
for i in range(0, total_rows, batch_size):
    end_idx = min(i + batch_size, total_rows)
    print(f"Processing batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size} (rows {i} to {end_idx-1})...")
    
    batch = df.iloc[i:end_idx].copy()
    batch['cleaned_text'] = batch[review_column].apply(clean_text)
    batch['tokens'] = batch['cleaned_text'].apply(tokenize_text)
    
    processed_df = pd.concat([processed_df, batch])
    
    print(f"Completed batch {i//batch_size + 1} in {time.time() - start_time:.2f} seconds")

df = processed_df

# Drop rows with empty tokenized text
df = df.dropna(subset=['tokens'])
print(f"After removing rows with empty tokens: {df.shape[0]} rows")

# Examples of cleaned and tokenized reviews
print("\nExample of cleaned and tokenized reviews:")
print(df[['cleaned_text', 'tokens']].head(3))

# Save the cleaned and tokenized data
output_file = 'cleaned_amazon_reviews.csv'
print(f"Saving to {output_file}...")
df.to_csv(output_file, index=False)

execution_time = time.time() - start_time
print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
print(f"Final dataset shape: {df.shape}")

