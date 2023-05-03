import argparse
import math
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import *
from transformers import AutoTokenizer

nltk.download('punkt')


import pandas as pd

def import_dataset(dataset_name: str, path: str) -> pd.Series:

    dataset_info = {
        'IMDB': {'column': 'review', 'format': 'csv'},
        'CVE': {'column': 'Description', 'format': 'csv'},
        'MITRE': {'column': 'text', 'format': 'csv'},
        'WikiText': {'column': 'body_text', 'format': 'csv'}
    }

    if dataset_name not in dataset_info:
        raise ValueError(f'Invalid dataset name: {dataset_name}. '
                         f'Accepted values are: {", ".join(dataset_info.keys())}.')
    
    info = dataset_info[dataset_name]

    try:
        if info['format'] == 'csv':
            if dataset_name == 'CVE':
                df = pd.read_csv(path, encoding='ISO-8859-1', low_memory=False)
            else:
                df = pd.read_csv(path)
            dataset = df[info['column']]
            print(f'{dataset_name} shape: {dataset.shape}')
        else:
            raise ValueError(f'Invalid format for dataset {dataset_name}. '
                             f'Accepted value is "csv".')
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found at path: {path}')

    return dataset



def extract_prominent_words(Q, Q_prime, top_k=5000, min_count=5):
    Q = re.sub(r'\b(?:[a-zA-Z]:)?(?:\\[a-zA-Z0-9_]+)+\\?', ' filepath ', Q)
    Q_prime = re.sub(r'\b(?:[a-zA-Z]:)?(?:\\[a-zA-Z0-9_]+)+\\?', ' filepath ', Q_prime)
    Q_tokens = nltk.word_tokenize(Q.lower())
    Q_prime_tokens = nltk.word_tokenize(Q_prime.lower())

    # Remove HTML tags, punctuation marks, and special characters
    Q = re.sub(r'<.*?>', '', Q)
    Q_prime = re.sub(r'<.*?>', '', Q_prime)
    Q = re.sub(r'[^\w\s]', '', Q)
    Q_prime = re.sub(r'[^\w\s]', '', Q_prime)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    Q_tokens = nltk.word_tokenize(Q)
    Q_tokens = [word for word in Q_tokens if word not in stop_words]
    Q_prime_tokens = nltk.word_tokenize(Q_prime)
    Q_prime_tokens = [word for word in Q_prime_tokens if word not in stop_words]

    Q_word_counts = Counter(Q_tokens)
    Q_prime_word_counts = Counter(Q_prime_tokens)

    Q_total_words = sum(Q_word_counts.values())
    Q_prime_total_words = sum(Q_prime_word_counts.values())
    p_Q = {word: count / Q_total_words for word, count in Q_word_counts.items()}
    p_Q_prime = {word: count / Q_prime_total_words for word, count in Q_prime_word_counts.items()}

    scores = []
    for word in p_Q:
        if Q_word_counts[word] >= min_count:
            if not any(char.isdigit() for char in word):
                if word not in p_Q_prime:
                    score = -p_Q[word] * math.log2(p_Q_prime.get(word, 1e-18))
                    scores.append((word, score))

    top_words = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return {word: score for word, score in top_words}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract prominent words')
    parser.add_argument('--dataset_Q', type=str, choices=['IMDB', 'CVE', 'MITRE', 'WikiText'],
                        help='Name of the first dataset to import and analyze')
    parser.add_argument('--path_Q', type=str, help='Path to the first dataset')
    parser.add_argument('--dataset_Q_prime', type=str, choices=['IMDB', 'CVE', 'MITRE', 'WikiText'],
                        help='Name of the second dataset to import and analyze')
    parser.add_argument('--path_Q_prime', type=str, help='Path to the second dataset')
    parser.add_argument('--top_k', type=int, default=5000, help='Number of prominent words to extract')
    parser.add_argument('--min_count', type=int, default=5, help='Minimum frequency count of words to consider')

    args = parser.parse_args()

    # Import the datasets
    dataset_Q = import_dataset(args.dataset_Q, args.path_Q)
    dataset_Q_prime = import_dataset(args.dataset_Q_prime, args.path_Q_prime)


    # Convert the datasets to strings
    dataset_Q_str = ' '.join(dataset_Q)
    dataset_Q_prime_str = ' '.join(dataset_Q_prime)

    # Extract prominent words from the query and query prompt
    prominent_words = extract_prominent_words(dataset_Q_str, dataset_Q_prime_str, args.top_k, args.min_count)

    # Print the top k prominent words and their scores
    print(f'Top {args.top_k} Prominent Words:')
    for word, score in prominent_words.items():
        print(f'{word}: {score:.4f}')

    # Visualize the distribution of prominent words in the dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    dataset_tokens = tokenizer((dataset_Q, dataset_Q_prime), padding=True, truncation=True, return_tensors='np')
    dataset_word_counts = Counter(np.ravel(dataset_tokens['input_ids']))
    dataset_word_counts = {tokenizer.convert_ids_to_tokens(id_): count for id_, count in dataset_word_counts.items()}

    top_words = sorted(prominent_words, key=prominent_words.get, reverse=True)[:10]
    top_words_counts = {word: dataset_word_counts.get(word, 0) for word in top_words}

    plt.bar(range(len(top_words_counts)), list(top_words_counts.values()), align='center')
    plt.xticks(range(len(top_words_counts)), list(top_words_counts.keys()), rotation=45)
    plt.title(f'Top {len(top_words_counts)} Prominent Words in {args.dataset_Q} and {args.dataset_Q_prime} datasets')
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.show()

    #To run: python cybersec_vocabulary.py --dataset_Q <DATASET_NAME> --path_Q <DATASET_PATH> --dataset_Q_prime <DATASET_NAME> --path_Q_prime <DATASET_PATH>
