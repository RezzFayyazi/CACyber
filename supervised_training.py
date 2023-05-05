import argparse
import os
import re
import random
import datetime
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import KFold
import nltk
from wordcloud import WordCloud, STOPWORDS
from transformers import BertTokenizer, RobertaTokenizer, RobertaForSequenceClassification
from transformers.models.bert import BertForSequenceClassification
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences


def import_dataset(path, column_name):
    df = pd.read_csv(path)
    column = df[column_name]
    print(f'{column_name} shape: {column.shape}')
    return column


def get_dataset_statistics(data, name):

    print(f'{name}:')
    print(data.info())
    print('--------------------------')
    print(data.describe())
    print('--------------------------')
    sentence_counts = data.value_counts()
    print(sentence_counts)
    print('--------------------------')
    max_length = max([len(nltk.word_tokenize(sentence)) for sentence in data])
    print(f'{name} max sentence length: {max_length}')


def cal_len(data):
    return len(data)

def calculate_word_frequency(data):

    count_words = data.str.split().apply(lambda z: cal_len(z))
    word_frequency = pd.DataFrame(count_words.value_counts(), columns=['Frequency'])
    word_frequency.index.name = 'Num. of words'
    return word_frequency


def plot_distribution(data, x_label, y_label, title):
    sns.displot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def display_word_cloud(data, color):
    plt.subplots(figsize=(10, 10))
    wc = WordCloud(stopwords=STOPWORDS, 
                   background_color="white", contour_width=2, contour_color=color,
                   max_words=2000, max_font_size=256,
                   random_state=42)
    wc.generate(' '.join(data))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def clean_text(data):    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    cleaned_data = []
    
    for sentence in data:
        # remove punctuation and convert to lowercase
        sentence = re.sub(r'[^\w\s]', '', sentence).lower()
        # remove stopwords
        sentence = ' '.join(word for word in sentence.split() if word not in stop_words)
        cleaned_data.append(sentence)
    
    return cleaned_data
