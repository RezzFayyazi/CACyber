# Contextual Understanding of Cybersecurity Exploits/Vulnerability Descriptions through Self-supervised Learning
This project aims to understand the context of cybersecurity descriptions by ranking the importance of each description with self-supervised metrics. There are three metrics that were tested in this project. 

1) Domain-sepcific word frequency 
The percentage of the occurrence of cybersecurity words in the sentences
2) Surprisal
To measure the unexpectedness of a word/phrase in the sentences
3) Rank Distance
Mask Language Modeling with the pre-trained language models, and measuring the distance of ground truth predictions between RoBERTa and SecureBERT.

## Usage
To be able to use the metrics, you need a domain-specific dictionary. You can run the "extract_dict.py" file, which is based on entropy to extract your desired dictionary. In this file, you need to pass two datasets. The first one should be the domain-specific dataset as you want to extract prominent and rare words in comparison with the second dataset, which should be a very general dataset containing general English words. For example, you can give the CVE descriptions and IMDB reviews to get the prominent words occuring in the CVEs. 
```python
python cybersec_vocabulary.py --dataset_Q <DATASET_NAME> --path_Q <DATASET_PATH> --dataset_Q_prime <DATASET_NAME> --path_Q_prime <DATASET_PATH>
```
### Code Example

Here is an example of how to use the project in code:

```python
import project_module




