import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import argparse

def wikipedia_crawl(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    vals = soup.find_all('div', attrs={"class":"mw-parser-output"})
    return vals

def extract_text(wiki):
    text_list = []
    for val in wiki:
        for child in val.children:
            if child.name == 'p':
                # Split paragraphs into sentences
                sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', child.text.strip())
                text_list.extend(sentences)
            elif child.name == 'ul':
                for li in child.find_all('li'):
                    # Split bullet point text into sentences
                    sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', li.text.strip())
                    for sentence in sentences:
                        if len(sentence.split()) >= 7:
                            text_list.append(sentence)
    return text_list

def crawl_wikipedia(query):
    # Search for articles with the given query
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': query,
        'srprop': 'size',
    }

    response = requests.get('https://en.wikipedia.org/w/api.php', params=params)
    data = response.json()

    # Loop over the search results and crawl each article
    text_list = []
    for result in data['query']['search']:
        url = 'https://en.wikipedia.org/wiki/' + result['title']
        wiki = wikipedia_crawl(url)
        text_list.extend(extract_text(wiki))

    # Create a pandas DataFrame and save to CSV
    df = pd.DataFrame({'text': text_list})
    df.to_csv('wiki_extracted.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crawl and extract text from Wikipedia articles.')
    parser.add_argument('query', metavar='QUERY', type=str, help='search query')
    args = parser.parse_args()
    crawl_wikipedia(args.query)

    #To run: python wikipedia_crawler.py "cybersecurity"