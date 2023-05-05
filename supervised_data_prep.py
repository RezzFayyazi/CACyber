import argparse
import os
import pandas as pd
import numpy as np
import nltk
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from keras.utils import to_categorical
from transformers import AutoTokenizer
from sklearn.metrics import *
from sklearn.model_selection import KFold

def import_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(data):
    '''
    Function to preprocess data by removing HTML syntax and punctuations
    '''
    html_tag = re.compile(r'<.*?>')
    punct_tag = re.compile(r'[^\w\s]')
    data = data.apply(lambda x: html_tag.sub(r'', x))
    data = data.apply(lambda x: punct_tag.sub(r'', x))
    return data

def analyze_data(data):
    '''
    Function to analyze the data by calculating word frequency and plotting distribution
    '''
    # Calculate word frequency
    word_count = data.str.split().apply(lambda x: len(x))
    print("Word count:", word_count)
    # Plot word frequency distribution
    sns.displot(word_count)
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.title("Word frequency distribution")
    plt.show()

def generate_wordcloud(data, color):
    '''
    Function to generate wordcloud
    '''
    plt.subplots(figsize=(10,10))
    wc = WordCloud(stopwords=STOPWORDS,
    background_color="white", contour_width=2, contour_color=color,
    max_words=2000, max_font_size=256,
    random_state=42)
    wc.generate(' '.join(data))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()

def merge_data(data1, data2):
    '''
    Function to merge two dataframes
    '''
    data1 = data1.rename('Description')
    data2 = data2.rename('Description')
    new_df = pd.concat([data1, pd.Series([0] * len(data1), index=data1.index, name='Technical')], axis=1)
    new_df2 = pd.concat([data2, pd.Series([1] * len(data2), index=data2.index, name='Technical')], axis=1)
    merged_df = pd.concat([new_df, new_df2], ignore_index=True, axis=0)
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)
    return merged_df

def get_args():
    '''
    Function to get command line arguments
    '''
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path1', required=True, help='Path of the first dataset')
    parser.add_argument('--path2', required=True, help='Path of the second dataset')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data1 = import_data(args.path1)
    data2 = import_data(args.path2)
    data1_desc = data1['Description']
    data2_desc = data2['Description']

    # Preprocess data
    data1_desc = preprocess_data(data1_desc)
    data2_desc = preprocess_data(data2_desc)

    # Generate word clouds and analyze data for both datasets
    print('Data1:')
    analyze_data(data1_desc)
    generate_wordcloud(data1_desc, 'red')
    print("----------------------------------")
    print('Data2:')
    analyze_data(data2_desc)
    generate_wordcloud(data2_desc, 'blue')

    # Merge datasets
    merged_df = merge_data(data1, data2)

    
# Evaluate model
# evaluate model on X_test and y_test using various metrics
# print the metrics
if __name__ == '__main__':
    main()