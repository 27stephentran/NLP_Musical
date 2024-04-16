import numpy as np
from keras.preprocessing.sequence import pad_sequences

class EmotionExtractor:
    def __init__(self, word2vec_model, emotion_model):
        self.word2vec_model = word2vec_model
        self.emotion_model = emotion_model
        self.emotions = ['sadness', 'anger', 'joy', 'fear', 'love', 'surprise']

    def extract_top_emotions(self, vectorize_sentence):
        
        probabilities = self.emotion_model.predict(np.array([vectorize_sentence]))
        
        # Get the indices of the top 3 emotions
        top_emotion_indices = np.argpartition(-probabilities[0], 3)[:3]
        
        # Get the names of the top 3 emotions
        top_emotions = [self.emotions[index] for index in top_emotion_indices]
        
        return top_emotions