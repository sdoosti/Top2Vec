"""
Created on August 12, 2024

@author: Shahryar Doosti (doosti@chapman.edu)

Prepares the text data by combining the sponsor info and video info without major cleaning
"""

import pandas as pd
import os, re
from nltk.corpus import stopwords
from collections import Counter
import datetime
import numpy as np

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
    
def combine_text(df1, df2):
    """
    Combine text data from two dataframes
    
    Args:
    df1: pd.DataFrame, first dataframe (including id and text column)
    df2: pd.DataFrame, second dataframe (including id and text column)
    
    Returns:
    pd.DataFrame, combined data
    """
    df1.columns = ['id','text']
    df2.columns = ['id','text']
    # set the id column as string
    df1['id'] = df1['id'].astype(str)
    df2['id'] = df2['id'].astype(str)
    return pd.concat([df1,df2],ignore_index=True)

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
    sponsor_info = load_data('sponsor_description.csv')
    sponsored_videos = load_data('sponsored_videos_data.csv')
    print('Done!')
    # get the text data
    print('Getting the text data...',end=' ')
    combined = combine_text(sponsor_info[['sponsor_id',"text"]],sponsored_videos[["new_id","text"]])
    docs = combined['text'].tolist()
    print('Done!')
    print(f'Total number of documents: {len(docs)}')
    # soft clean the text data
    print('Soft cleaning the text data...',end=' ')
    docs = soft_clean(docs)
    print('Done!')
    # save the processed text data
    print('Saving the processed text data...')
    output = os.path.join(DATA_PATH,f'vca_text_{today_str}.txt')
    with open(output,'w') as f:
        for doc in docs:
            f.write(doc+'\n')
    print(f'Text data is saved in {output}')