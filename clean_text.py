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
        return data.fillna('')[cols].lower().values.tolist()
    else:
        # modifying the topics column (assuming it is a list of topics with ";" delimiter)
        topics = data[cols[1]].fillna('').lower().values.tolist()
        topics = ["topics are "+" ".join(topic.split(';')) for topic in topics if topic != '']
        # merging topics column with text column
        docs = data[cols[0]].fillna('').lower().values.tolist()
        return [doc + ' ' + topic for doc,topic in zip(docs,topics)]

def soft_clean(docs):
    """
    Soft clean the text data: removing new lines, single quotes, extra spaces, ...
    
    Args:
    docs: list of str, list of documents
    
    Returns:
    list of str, cleaned documents
    """
    # Remove new line characters
    processed_docs = [re.sub('\s+', ' ', sent) for sent in docs]
                
    # Remove 's
    processed_docs = [re.sub("(\'s)","",sent) for sent in processed_docs]
                
    # Remove distracting single quotes
    processed_docs = [re.sub("\'", "", sent) for sent in processed_docs]

    # Remove extended stop words
    data = [' '.join([w for w in sent.split() if not w.lower() in exstopwords]) for sent in data]

    docs = [x.strip() for x in data] 
    doc_ids = pooled.video_id.tolist()  

    # combined stopwords
    swlist = list(STOP_WORDS)
    swlist.extend(extended_stopwords)
    swlist.extend(stopwords.words('english'))
    STOP_WORDS = set(swlist)
    
    return docs




print('Loading ...')         



print('Basic cleaning ...')         
data = pooled.text.values.tolist()

# not removing emails, hashtags, and urls as they will be removed anyway
# unless they are important

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
               
# Remove 's
data = [re.sub("(\'s)","",sent) for sent in data]
               
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

# Remove extended stop words
exstopwords = set(extended_stopwords)
data = [' '.join([w for w in sent.split() if not w.lower() in exstopwords]) for sent in data]

docs = [x.strip() for x in data] 
doc_ids = pooled.video_id.tolist()  

# combined stopwords
swlist = list(STOP_WORDS)
swlist.extend(extended_stopwords)
swlist.extend(stopwords.words('english'))
STOP_WORDS = set(swlist)

nlp = spacy.load('en')        

print('Processing ...')         
processed_docs = []    
for doc in nlp.pipe(docs, n_threads=8, batch_size=100):
    # Process document using Spacy NLP pipeline.
    
    ents = doc.ents  # Named entities.

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    doc = [token.lemma_.lower().strip() for token in doc if token.is_alpha and not token.is_stop  and token.lemma_ != '-PRON-']

    # Remove common words from a stopword list.
    doc = [token for token in doc if token not in STOP_WORDS]

    # Add named entities, but only if they are a compound of more than word.
    doc.extend([str(entity) for entity in ents if len(entity) > 1])
    
    processed_docs.append(doc)
    
docs = processed_docs.copy()
#del processed_docs

print('Saving the text ...')         
with open('/homes/shahryar/Desktop/Tubular/top_creators/processed_docs_jan202019.txt','w') as f:    
    for doc in docs:
        f.write(','.join(filter(lambda x: x not in ['',' ','[]','[ ]'],doc))+'\n')
        

print('Cleaning Process Completed!')