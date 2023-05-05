# Contextual Understanding of Cybersecurity Exploits/Vulnerability Descriptions through Self-supervised Learning
This project aims to understand the context of cybersecurity descriptions by ranking the importance of each description with self-supervised metrics. There are three metrics that were tested in this project. These metrics are to measure the degree of technicality of cybersecurity-specialized sentences.

1) Domain-sepcific word frequency: The percentage of the occurrence of cybersecurity words in the sentences
2) Surprisal: To measure the unexpectedness of a word/phrase in the sentences
3) Rank Distance: Mask Language Modeling (MLM) with the pre-trained language models, namely RoBERTa and SecureBERT, to measure the distance of ground truth predictions between each LM.

## Usage
To be able to use the metrics, you need a domain-specific dictionary. You can run the "extract_dict.py" file, which is based on entropy to extract your desired dictionary. In this file, you need to pass two datasets. The first one should be the domain-specific dataset as you want to extract prominent and rare words in comparison with the second dataset, which should be a very general dataset containing general English words. For example, you can give the CVE descriptions and IMDB reviews to get the prominent words occuring in the CVEs. Here is the way of how to run it.
```python
python extract_dict.py --dataset_Q <DATASET_NAME> --path_Q <DATASET_PATH> --dataset_Q_prime <DATASET_NAME> --path_Q_prime <DATASET_PATH>
```

After you have your dictionary, you can now use the metrics to evaluate the performance of cybersecurity sentences with the "main.py" file. Here is the way to run it:

```python
python main.py data_file_path.csv dictionary_file_path.csv --metric metric_name --normalize normalization_type
```
The "eval.ipynb" notebook is the consolidated code to play around with different metrics and evaluating your concatenated dataset.

## Additional
The "eval_supervised_models.py" file is to evaluate the classification performance of cybersecurity-specialized sentences to measure the accuracy of classifiers in distinguishing the cybersecurity vs. generic sentences. To run:
```python
python script_name.py model_path data_path
```

You can also train your classifier by passing the arguments. Here is the way to run it:
```python
python train.py --train_data_path /path/to/training/data --output_path /path/to/save/trained/model --batch_size 64 --learning_rate 0.0001 --num_epochs 20 --print_freq 50
```
