"""
Created on Jun 19 2024

@author: Shahryar Doosti (doosti@chapman.edu)

Clean the data and saves the text
"""

import pandas as pd
import os, re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS 
from nltk.corpus import stopwords
from collections import Counter
import sys
import datetime

today = datetime.date.today()
today_str = today.strftime('%Y-%m-%d')

# setting path to the parent directory of the file
PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PATH,'data')

def load_data(filename):
    """
    Load data from the data directory
    
    Args:
    filename: str, name of the file to load from the 'data' directory

    Returns:
    pd.DataFrame, loaded data
    """
    if os.path.exists(os.path.join(DATA_PATH,filename)):
        return pd.read_csv(os.path.join(DATA_PATH,filename)) # ,encoding='iso-8859-1'
    elif os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        raise FileNotFoundError('File not found!')

def get_text(data, cols=['text','topics']):
    """
    Get text data from the data
    
    Args:
    data: pd.DataFrame, data containing text and topics columns
    cols: list of str, columns to get the text data from
            text: str, text data
            topics: str, topics data
    
    Returns:
    list of str, text data
    """
    if len(cols)==1:
        return data.fillna('')[cols].str.lower().values.tolist()
    else:
        # modifying the topics column (assuming it is a list of topics with ";" delimiter)
        topics = data[cols[1]].fillna('').str.lower().values.tolist()
        # add "topics are" to the beginning of each topic if the topic is not empty
        topics = [topic if topic=='' else 'topics are '+" ".join(topic.split(';')) for topic in topics]
        # merging topics and description column with text column
        docs = data[cols[0]].fillna('').str.lower().values.tolist()
        return [doc + ' ' + topic  for doc,topic in zip(docs,topics)]

def soft_clean(docs):
    """
    Soft clean the text data: removing new lines, single quotes, extra spaces, ...
    
    Args:
    docs: list of str, list of documents
    
    Returns:
    list of str, cleaned documents
    """
    processed_docs = [x.strip() for x in docs]

    # Remove new line characters
    processed_docs = [re.sub('\s+', ' ', sent) for sent in processed_docs]
                
    # Remove 's
    processed_docs = [re.sub("(\'s)","",sent) for sent in processed_docs]
                
    # Remove distracting single quotes
    processed_docs = [re.sub("\'", "", sent) for sent in processed_docs]

    return processed_docs

nlp = spacy.load('en_core_web_trf') 

def preprocessing(text, nlp):
    # Process document using Spacy NLP pipeline.
    doc = nlp(text)
    ents = doc.ents  # Named entities.

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    tokens = [token.lemma_.lower().strip() for token in doc if token.is_alpha and not token.is_stop and token.lemma_ != '-PRON-' and not token.is_punct]

    # Add named entities, but only if they are a compound of more than word.
    #tokens.extend([str(entity) for entity in ents if len(entity) > 1])
    
    return tokens


if __file__ == '__main__':
    # load the data
    print('Loading the data...', end=' ')
    data = load_data('data.csv')
    print('Done!')
    # get the text data
    print('Getting the text data...',end=' ')
    docs = get_text(data)
    print('Done!')
    # soft clean the text data
    print('Soft cleaning the text data...',end=' ')
    docs = soft_clean(docs)
    print('Done!')
    # preprocess the text data
    print(f"Preprocessing the text data using Spacy NLP pipeline ({len(docs)} documents)...",end=' ')
    #processed_docs = [preprocessing(doc, nlp) for doc in docs]
    processed_docs = []    
    for doc in nlp.pipe(docs):
        
        #ents = doc.ents  # Named entities.

        # Keep only words (no numbers, no punctuation).
        # Lemmatize tokens, remove punctuation and remove stopwords.
        doc = [token.lemma_.lower().strip() for token in doc if token.is_alpha and not token.is_stop and token.lemma_ != '-PRON-' and not token.is_punct]

        # Add named entities, but only if they are a compound of more than word.
        #doc.extend([str(entity) for entity in ents if len(entity) > 1])
        
        processed_docs.append(doc)    
    print('Done!')
    # save the processed text data
    print('Saving the processed text data...')
    with open('processed_text__{today_str}.txt','w') as f:
        for doc in processed_docs:
            #f.write(' '.join(doc)+'\n')
            f.write(','.join(filter(lambda x: x not in ['',' ','[]','[ ]'],doc))+'\n')
    print('Text data is saved in processed_text.txt')