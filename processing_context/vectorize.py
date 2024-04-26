import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api


# Function to vectorize a sentence
def vectorize_sentence(sentence):
    word2vec_model = api.load("word2vec-google-news-300")
    # Tokenize the sentence
    tokens = sentence.split()
    
    # Initialize an empty list to store word vectors
    word_vectors = []
    
    # Iterate through tokens and get their vectors
    for token in tokens:
        if token in word2vec_model:
            word_vectors.append(word2vec_model[token])
    
    # If no word vectors found, return None
    if not word_vectors:
        return None
    
    # Average the word vectors to get the sentence vector
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector

