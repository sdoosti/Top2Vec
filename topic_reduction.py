# This file loads a trained top2vec model and reduce the number of topics and saves the hierarchy of topics
from top2vec import Top2Vec
import os
import pandas as pd
import numpy as np
import pickle
import logging
import datetime

today = datetime.date.today()
today_str = today.strftime('%Y-%m-%d')

PATH = "/home/doosti@chapman.edu/projects/Facebook/top2vec/"
DATA_PATH = os.path.join(PATH,'data')

model_name = "top2vec_deeplearn_distiluse_notoken_2024-06-27.model"
output = model_name.replace('.model','_reduced.model')

OUTPUT = os.path.join(DATA_PATH,output)

model_path = os.path.join(DATA_PATH,model_name)

logging.basicConfig(filename=OUTPUT.replace("model","log"), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    

model = Top2Vec.load(model_path)
logging.info('Model is loaded.')

logging.info('Reducing the number of topics to 20.')
hierarchy = model.hierarchical_topic_reduction(20)
logging.info('Hierarchy is created.')

# saving model
model.save(OUTPUT)
logging.info('Model is saved.')

# saving hierarchy
with open(OUTPUT.replace("model","hierarchy"), 'wb') as f:
    pickle.dump(hierarchy, f)
logging.info('Hierarchy is saved.')



