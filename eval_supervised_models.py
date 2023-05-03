import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt


def load_model(path: str, pretrained_model: str, data: str):
    """
    Loads the saved RoBERTa model from the specified path, tokenizes the input sequences, 
    passes the encoded inputs through the model to obtain predictions, converts the logits 
    to probabilities, and returns the predicted labels and probabilities.
    """
    # Create a new RoBERTa model object and load the saved weights into it
    model = RobertaForSequenceClassification.from_pretrained(path)

    # Set the model to evaluation mode and disable gradient calculations
    model.eval()
    torch.set_grad_enabled(False)

    # Tokenize the input sequences
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
    encoded_inputs = tokenizer(data, padding=True, truncation=True, return_tensors='pt')

    # Pass the encoded inputs through the model to obtain predictions
    with torch.no_grad():
        outputs = model(encoded_inputs['input_ids'], attention_mask=encoded_inputs['attention_mask'])

    logits = outputs.logits

    # Convert the logits to probabilities and obtain the predicted labels
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    return preds, probs


def visualize_predictions(preds):
    """
    Visualizes the distribution of predicted labels.
    """
    sns.set(style='darkgrid')
    ax = sns.countplot(x=preds.numpy())
    ax.set(xlabel='Predicted Labels', ylabel='Count')
    plt.show()


def main():
    """
    Main function to load and use the RoBERTa model on the given data.
    """
    parser = argparse.ArgumentParser(description='Load and use a saved RoBERTa model.')
    parser.add_argument('model_path', type=str, help='path to the saved RoBERTa model')
    parser.add_argument('data_path', type=str, help='path to the input data file')
    args = parser.parse_args()

    # Read the input data file
    data = pd.read_csv(args.data_path)

    # Extract the text data from the input data file
    sentences = data['text'].values
    all_texts = []
    for sent in sentences:
        all_texts.append(sent)

    # Load the saved RoBERTa model and obtain the predicted labels and probabilities
    preds, probs = load_model(args.model_path, "ehsanaghaei/SecureBERT", all_texts)

    # Print the predicted labels and probabilities
    print("Predicted labels: ", preds)
    print("Probabilities: ", probs)

    # Visualize the distribution of predicted labels
    visualize_predictions(preds)


if __name__ == '__main__':
    main()
    #To run: python script_name.py model_path data_path
    #E.g. python script.py ./SecureBERT_technical_vs_generic_CVE_and_MITRE/ comp_sec_wiki.csv
