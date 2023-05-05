import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from normalization import *
import argparse
import transformers
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from scipy.stats.mstats import winsorize
import re
import random
import math

def read_data(data_path):
    """
    Reads the data from the given CSV file path and returns a Pandas DataFrame object.
    """
    df = pd.read_csv(data_path)
    return df


def read_dictionary(dictionary_path):
    """
    Reads the cybersecurity dictionary from the given CSV file path and returns a list of cybersecurity words.
    """
    dictionary = pd.read_csv(dictionary_path)['Dictionary'].dropna().astype(str).apply(lambda x: x.lower()).values
    stop_words = set(stopwords.words('english'))
    cybersecurity_words = [word.lower() for word in dictionary if word.lower() not in stop_words]
    return cybersecurity_words


def calculate_frequency(sentences, cybersecurity_words):
    """
    Calculates the frequency of cybersecurity words in each sentence of the given list of sentences.
    Returns a list of frequency values for each sentence.
    """
    num_words_per_sentence = []
    for sentence in sentences:
        words = sentence.lower().split()
        num_of_words = len(words)
        num_cybersecurity_words = sum(1 for word in words if word in cybersecurity_words)
        percentage_of_words = (num_cybersecurity_words / num_of_words) * 100
        num_words_per_sentence.append(percentage_of_words)
    return num_words_per_sentence

def surprisal_cybersecurity_words(sentences, cybersecurity_words, model_name='bert-base-uncased'):
    # load the language model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForMaskedLM.from_pretrained(model_name)  

    top_words_dict = {}
    avg_surprisal_sentences = []
    for sentence in sentences:  
        surprisal_sentences = [] 
        cyber_words = [] 
        count = 0
        tokens = tokenizer.encode(sentence, return_tensors='pt')  # encode the sentence into a tensor of tokens
        logits = model(tokens).logits  # get the logits from the model
        probs = torch.softmax(logits[0, :-1], dim=1)  # calculate the probabilities of the tokens
        
        for i, token in enumerate(tokens[0, 1:]):
            word = tokenizer.decode(token.unsqueeze(0)).lower()  # decode the token and convert to lowercase
            if word in cybersecurity_words:  # if the token is a cybersecurity word
                count += 1
                surprisal = -torch.log2(probs[i, token]).item()  # calculate the surprisal value of the token
                surprisal_sentences.append(surprisal)  # add the surprisal value to the list
                cyber_words.append((surprisal, word))  # add the word and its surprisal value to the list
            else:
                continue  # if the token is not a cybersecurity word, skip it
        
        if count > 0:
            avg_surprisal = sum(surprisal_sentences)/count 
        else:
            avg_surprisal = 0 
        
        avg_surprisal_sentences.append(avg_surprisal)    
        top_words_dict[sentence] = cyber_words  

    return top_words_dict, avg_surprisal_sentences 



def get_sentences_with_higher_rank(dataset, dictionary, model_roberta, model_securebert, tokenizer_roberta, tokenizer_securebert):
    l1_distances = []
    sentences = []
    for sentence_idx, sentence in enumerate(dataset['Sentence']):
        # Check if any words in the sentence are in the dictionary and mask them
        words = sentence.lower().split()
        masked_indices = []
        masked_words = []
        for i, word in enumerate(words):
            if word in dictionary:
                words[i] = '<mask>'
                masked_indices.append(i)
                masked_words.append(word)
        masked_sentence = ' '.join(words)
        # Tokenize the sentence using both Roberta and SecureBERT tokenizers
        tokenized_sentence_roberta = tokenizer_securebert.encode(masked_sentence, add_special_tokens=True)
        tokenized_sentence_securebert = tokenizer_securebert.encode(masked_sentence, add_special_tokens=True)
        # If the lengths of the tokenized sentences are not the same, skip to the next sentence
        if len(tokenized_sentence_roberta) != len(tokenized_sentence_securebert):
            print('yes')
            continue

        # If no words in the sentence are in the dictionary, skip to the next sentence
        if len(masked_indices) == 0:
            l1_distances.append(0)
            #print(sentence)
            continue

        # Randomly sample up to 3 masked indices or all masked indices if there are fewer than 3
        try:
            random_indices = random.sample(masked_indices, min(3, len(masked_indices)))
        except ValueError:
            random_indices = masked_indices
            
        for masked_index in masked_indices:
            if masked_index in random_indices:
                if masked_index < len(tokenized_sentence_securebert):
                    tokenized_sentence_roberta[masked_index] = tokenizer_roberta.mask_token_id
                    tokenized_sentence_securebert[masked_index] = tokenizer_securebert.mask_token_id
                    words[masked_index] = masked_words[random_indices.index(masked_index)]
            else:
                words[masked_index] = masked_words[masked_indices.index(masked_index)]
        
        # Convert the tokenized sentences to tensors and move them to the appropriate device
        tokens_tensor_roberta = torch.tensor([tokenized_sentence_roberta]).to(model_roberta.device)
        tokens_tensor_securebert = torch.tensor([tokenized_sentence_securebert]).to(model_securebert.device)

        # Compute predictions for the masked indices using both Roberta and SecureBERT models
        with torch.no_grad():
            outputs_roberta = model_roberta(tokens_tensor_roberta)
            predictions_roberta = outputs_roberta[0][0, [i for i in masked_indices]].softmax(-1)

        with torch.no_grad():
            outputs_securebert = model_securebert(tokens_tensor_securebert)
            predictions_securebert = outputs_securebert[0][0, [i for i in masked_indices]].softmax(-1)

        # Get the ground truth ranks for each masked word
        gt_ranks = []
        for i, masked_index in enumerate(random_indices):
            gt_word = masked_words[i]
            gt_prob_roberta = predictions_roberta[i, tokenizer_roberta.convert_tokens_to_ids("Ġ" + gt_word)].item()
            gt_prob_securebert = predictions_securebert[i, tokenizer_securebert.convert_tokens_to_ids("Ġ" + gt_word)].item()
            gt_rank_roberta = (predictions_roberta[i] > gt_prob_roberta).sum().item() + 1
            gt_rank_securebert = (predictions_securebert[i] > gt_prob_securebert).sum().item() + 1
            gt_ranks.append((gt_rank_roberta, gt_rank_securebert))

        # 1. Determine which masked words SecureBERT performed better on
        better_words = []
        for i, (rank_roberta, rank_securebert) in enumerate(gt_ranks):
            if rank_securebert < rank_roberta:
                better_words.append(i)

        # 2. If no words were better for SecureBERT, append 0 to the list of distances and continue to the next sentence
        if len(better_words) == 0:
            l1_distances.append(0)
        else:
        # 3. Compute the L1 distance between the Roberta and SecureBERT ranks for the better words
            l1_distance = 0
            for i in better_words:
                #if gt_ranks[i][0] < 500:
                #l1_distance += abs(gt_ranks[i][0] - gt_ranks[i][1])
                l1_distance += abs(np.log(gt_ranks[i][0]) - np.log(gt_ranks[i][1]))

            avg_l1_distance = l1_distance / len(better_words)
            # 4. Append the log l1 distance to the list of distances
            l1_distances.append(avg_l1_distance)

    return l1_distances



def plot_histograms(data, title):
    """
    Plots a histogram for the given data with the given title.
    """
    fig, ax = plt.subplots()
    ax.hist(data, bins=20)
    ax.set_title(title)
    plt.show()


def main(args):

    df = read_data(args.data_path)
    sentences = df['Sentence'].dropna().values.astype(str).tolist()
    cybersecurity_words = read_dictionary(args.dictionary_path)

    # Determine the metric to use
    if args.metric == "frequency":
        num_words_per_sentence = calculate_frequency(sentences, cybersecurity_words)
        data = num_words_per_sentence
        title = "Cybersecurity Word Frequency"
    elif args.metric == "surprisal":
        top_w, avg_surprisal = surprisal_cybersecurity_words(sentences, cybersecurity_words, model_name= 'bert-base-uncased')
        countr=0
        for item in avg_surprisal:
            if item == 0:
                countr += 1
        print(countr)
        data = avg_surprisal
        title = "Average Surprisal of Cybersecurity Words"
    elif args.metric == "rank":
        model_roberta = RobertaForMaskedLM.from_pretrained('roberta-base')
        model_securebert = RobertaForMaskedLM.from_pretrained('ehsanaghaei/SecureBERT')
        tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer_securebert = RobertaTokenizer.from_pretrained('ehsanaghaei/SecureBERT')
        distances = get_sentences_with_higher_rank(df, cybersecurity_words, model_roberta, model_securebert, tokenizer_securebert, tokenizer_securebert)
        data = distances
        title = "Average Rank Distance of Cybersecurity Words"
    # Plot the histogram for the original data
    plot_histograms(data, title)

    # Normalize the data
    if args.normalize == "mad":
        normalized_data = mad_normalize(data)
        title = "MAD Normalized Data"
    elif args.normalize == "iqr":
        normalized_data = iqr_normalize(data)
        title = "IQR Normalized Data"
    elif args.normalize == "winsorize":
        normalized_data = winsorize_normalize(data, lower_bound=0.05, upper_bound=0.05)
        title = "Winsorized Normalized Data"
    elif args.normalize == "logarithmic":
        normalized_data = normalize_data_with_logarithm(data)
        title = "Logarithm Normalized Data"
    elif args.normalize == "regular":
        normalized_data = normalize_data(data)
        title = "Regular Normalized Data"
    elif args.normalize == "none":
        normalized_data = data
        title = title + " (Unnormalized)"

    # Plot the histogram for the normalized data
    plot_histograms(normalized_data, title)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain-specific word frequency metric')
    parser.add_argument('data_path', type=str, help='The path to the CSV file containing the data')
    parser.add_argument('dictionary_path', type=str, help='The path to the CSV file containing the cybersecurity dictionary')
    parser.add_argument('--metric', type=str, choices=['frequency', 'surprisal', 'rank'], default='frequency', help='The type of metric to use')
    parser.add_argument('--normalize', type=str, choices=['mad', 'iqr', 'winsorize','logarithmic', 'regular',  'none'], default='none', help='The type of normalization to apply to the data')
    args = parser.parse_args()
    main(args)
    # To run: python metrics.py data_file_path.csv dictionary_file_path.csv --metric metric_name --normalize normalization_type
