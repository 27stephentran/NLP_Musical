import json
import numpy as np
from processing_context.vectorize import vectorize_sentence
from processing_context.get_mood import EmotionExtractor
import gensim.downloader as api
from keras.models import model_from_json


# Load the songs JSON file
<<<<<<< HEAD
with open("data/songs_with_emotions.json", "r", encoding="utf-8") as f:
=======
with open("data/test_songs.json", "r", encoding="utf-8") as f:
>>>>>>> 3e70dfd5ecc30d2b0267202abf7e5ee39f32788c
    songs = json.load(f)


word2vec_model = api.load("word2vec-google-news-300")
with open("custom_model/my_model.json", "r") as json_file:    
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the word2vec model
word2vec_model = api.load("word2vec-google-news-300")

# Initialize the EmotionExtractor
extractor = EmotionExtractor(word2vec_model, loaded_model)

# Iterate through each song and extract emotions
for song_id, song_info in songs.items():
    lyric = song_info["lyric"]
    # Vectorize the lyric
    word_vectors = [word2vec_model[word] for word in lyric.split() if word in word2vec_model]

    # Average the word vectors to get sentence vector
    sentence_vector = np.mean(word_vectors, axis=0)
    # Extract top 3 emotions
    top_emotions = extractor.extract_top_emotions(sentence_vector)
    # Add top 3 emotions to the song info
    songs[song_id]["top_emotions"] = top_emotions

# Save the updated JSON file
<<<<<<< HEAD
with open("data/songs_with_emotions.json", "w") as f:
=======
with open("songs_with_emotions.json", "w") as f:
>>>>>>> 3e70dfd5ecc30d2b0267202abf7e5ee39f32788c
    json.dump(songs, f, indent=4)
