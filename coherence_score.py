import gensim
from top2vec import Top2Vec
from gensim.models.coherencemodel import CoherenceModel
import os
import argparse

PATH = "/home/doosti@chapman.edu/projects/Facebook/top2vec/"
DATA_PATH = os.path.join(PATH,'data')
model_name = "top2vec_deeplearn_doc2vec_notoken_2024-06-21.model"

# get model name from arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_name', help='Top2Vec Model Name',type=str, default=model_name)
    args = parser.parse_args()
    return args

def coherence_score(model_path, num_topics=None):
    """
    Calculate coherence score of a Top2Vec model
    :param model_path: str, path of the model file
    :param num_topics: int, number of topics to consider, if None all topics will be considered
    :return: float, coherence score
    """
    # Load Top2Vec model
    model = Top2Vec.load(model_path)
    # Get the topic words
    if num_topics is not None:
        topics_words, word_scores, topic_numbs = model.get_topics(num_topics=num_topics)
    else:
        topics_words, word_scores, topic_numbs = model.get_topics()
    # Create a Dictionary and Corpus for Gensim (required for coherence model)
    dictionary = gensim.corpora.Dictionary(topics_words)
    corpus = [dictionary.doc2bow(doc) for doc in topics_words]
    # Build Coherence Model
    cm = CoherenceModel(model=None, topics=topics_words, texts=model.documents, dictionary=dictionary, coherence='c_v')
    coherence_score = cm.get_coherence()
    print("Coherence Score (C_V):", coherence_score)
    return coherence_score

if __name__ == "__main__":
    args = get_args()
    model_name = args.model_name
    model_path = os.path.join(DATA_PATH, model_name)
    coherence_score(model_path)