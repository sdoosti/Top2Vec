"""
Created on Jun 19 2024

@author: Shahryar Doosti (doosti@chapman.edu)

Prepares the text data without major cleaning
"""

import pandas as pd
import os, re
from nltk.corpus import stopwords
from collections import Counter
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
        return pd.read_csv(os.path.join(DATA_PATH,filename), low_memory=False) # ,encoding='iso-8859-1'
    elif os.path.exists(filename):
        return pd.read_csv(filename, low_memory=False) # ,encoding='iso-8859-1'
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

    # Remove punctuation
    processed_docs = [re.sub(r'[^\w\s]','',sent) for sent in processed_docs]

    # Remove urls
    processed_docs = [re.sub(r'http\S+','',sent) for sent in processed_docs]

    # Remove stopwords
    #stop_words = stopwords.words('english')
    #processed_docs = [' '.join([word for word in sent.split() if word not in stop_words]) for sent in processed_docs]

    # Remove hashtags
    processed_docs = [re.sub(r'#\w+','',sent) for sent in processed_docs]

    # Remove extra spaces
    processed_docs = [re.sub(" +", " ", sent) for sent in processed_docs]

    return processed_docs


if __name__ == '__main__':
    # load the data
    print('Loading the data...', end=' ')
    data = load_data('data_all_text.csv')
    print('Done!')
    # get the text data
    print('Getting the text data...',end=' ')
    docs = get_text(data)
    print('Done!')
    # soft clean the text data
    print('Soft cleaning the text data...',end=' ')
    docs = soft_clean(docs)
    print('Done!')
    # save the processed text data
    print('Saving the processed text data...')
    output = os.path.join(DATA_PATH,f'not_processed_text_{today_str}.txt')
    with open(output,'w') as f:
        for doc in docs:
            f.write(doc+'\n')
    print(f'Text data is saved in {output}')