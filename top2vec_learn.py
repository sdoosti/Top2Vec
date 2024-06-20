"""
This file is to learn a new top2vec model from the text data.
It only works on the (new) preprocessed data.
"""
import os
import pandas as pd
import numpy as np
from top2vec import Top2Vec
import sys
import re
import datetime
import argparse
import logging

today = datetime.date.today()
today_str = today.strftime('%Y-%m-%d')

# setting path to the parent directory of the file
PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PATH,'data')

def get_args():
    """
    Get the command line arguments
    """
    parser = argparse.ArgumentParser(description='Train the top2vec topic models')
    parser.add_argument('-f','--file', help='Text data', default=os.path.join(DATA_PATH,"processed_text_2024-06-20.txt"))
    parser.add_argument('-e','--embedding', help='Embedding model', default='doc2vec')
    parser.add_argument('-s','--speed', help='Speed of the model', default='learn')
    parser.add_argument('-o','--output', help='The file to write the model', default=os.path.join(DATA_PATH,f'top2vec_{today_str}.model'))
    args = parser.parse_args()
    return args

def check_args(args):
    """
    Check the command line arguments
    """
    assert args.embedding in ['doc2vec','universal-sentence-encoder', 'distiluse-base-multilingual-cased-v2', 'distilbert-base-nli-mean-tokens'], 'Embedding model should be either doc2vec, universal-sentence-encoder, distiluse-base-multilingual-cased-v2 or distilbert-base-nli-mean-tokens'
    assert args.speed in ['fast-learn', 'learn', 'deep-learn'], 'Speed should be either fast-learn, learn or deep-learn'
    assert args.output.endswith('.model'), 'Output file should be a .model file'

def print_model_setup(args):
    """
    Print the model setup
    """
    print('-'*50)
    print('Model setup:')
    print(f"Model type: tokens")
    print(f"Speed: {args.speed}")
    print(f"Embedding model: {args.embedding}")
    print('-'*50)

def load_data(filename):
    """
    Load data from the data directory
    
    Args:
    filename: str, name of the file to load from the 'data' directory

    Returns:
    list of documents, loaded data
    """
    with open(filename,"r", encoding="utf-8") as f:
        processed_docs = f.readlines()
    processed_docs = [re.sub("\d+", "", x.strip()).split(',') for x in processed_docs]
    logging.info(f"Data is loaded from {filename}")
    logging.info(f"Number of documents: {len(processed_docs)}")
    #docs = [" ".join(x) for x in processed_docs if len(x) > 5]
    docs = [" ".join(x) for x in processed_docs]
    return docs

def save_model(model, filename):
    """
    Save the model to a file
    
    Args:
    model: Top2Vec, the model to save
    filename: str, the file to save the model to
    """
    model.save(filename)
    print("Top2Vec model is saved.")

def main():
    """
    Main function
    """
    args = get_args()
    check_args(args)
    print_model_setup(args)
    
    # Start the logging file
    logging.basicConfig(filename=f'top2vec_{today_str}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # log the model setup
    logging.info('Model setup:')
    logging.info(f"Model type: tokens")
    logging.info(f"Speed: {args.speed}")
    logging.info(f"Embedding model: {args.embedding}")

    # Load the data
    docs = load_data(os.path.join(args.file))

    # Train the model
    logging.info('Training the model...')
    model = Top2Vec(documents=docs, speed=args.speed, workers=8, embedding_model=args.embedding, keep_documents=True)
    logging.info('Model is trained.')
    save_model(model, args.output)
    logging.info('Model is saved.')

if __name__ == '__main__':
    main() 