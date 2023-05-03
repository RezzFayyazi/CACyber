import argparse
import math
import os
import random
import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.stats.mstats import winsorize

nltk.download('punkt')
warnings.filterwarnings('ignore')


def process_dataset(dataset_path):
    """
    Process a single dataset by reading the input file, filtering and preprocessing the sentences,
    and adding them to a DataFrame along with their labels.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} does not exist")

    # Read the input CSV file into a pandas DataFrame
    dataset_df = pd.read_csv(dataset_path)

    # Rename the columns to "Sentence" and "Label"
    dataset_df = dataset_df.rename(columns={'text': 'Sentence', 'label': 'Label'})

    # Filter sentences with length less than 200
    dataset_df = dataset_df[dataset_df['Sentence'].apply(lambda x: len(x) < 200)]

    return dataset_df


def combine_datasets(datasets, num_chunks_list):
    """
    Combine multiple processed datasets into a single DataFrame by randomly selecting one chunk
    from each dataset.
    """
    combined_dfs = []
    for i, df in enumerate(datasets):
        num_chunks = num_chunks_list[i]
        chunks = np.array_split(df, num_chunks)
        random_chunk = random.choice(chunks)
        combined_dfs.append(random_chunk)

    combined_df = pd.concat(combined_dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    return combined_df


def write_to_csv(df, output_file):
    """
    Write the DataFrame to a CSV file.
    """
    df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Combine and preprocess multiple datasets')
    parser.add_argument('--datasets', nargs='+', required=True, help='list of paths to the input CSV files')
    parser.add_argument('--output', required=True, help='path to the output CSV file')
    parser.add_argument('--num-chunks', nargs='+', type=int, required=True, help='list of numbers of chunks to split the datasets into')
    args = parser.parse_args()

    datasets = []
    for dataset_path in args.datasets:
        dataset_df = process_dataset(dataset_path)
        datasets.append(dataset_df)

    combined_df = combine_datasets(datasets, args.num_chunks)
    write_to_csv(combined_df, args.output)


if __name__ == '__main__':
    main()
    # To run: python preprocess_datasets.py --datasets path/to/dataset1.csv path/to/dataset2.csv --output path/to/output.csv --num-chunks 3 4

