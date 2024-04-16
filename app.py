import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from processing_context.clean_context import preprocess_text_safe
from processing_context.get_mood import EmotionExtractor
import gensim.downloader as api
import numpy as np
from keras.models import model_from_json
import json

with open("custom_model/my_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

word2vec_model = api.load("word2vec-google-news-300")

saved_status = None  

def get_emo(word2vec_model, loaded_model, sentence):
    extractor = EmotionExtractor(word2vec_model, loaded_model)
    top_emotions = extractor.extract_top_emotions(sentence)
    return top_emotions

def submit_status():
    global saved_status
    status = status_entry.get()
    if status:
        sentence = status
        
        # Vectorize the sentence using the word2vec model
        word_vectors = [word2vec_model[word] for word in sentence.split() if word in word2vec_model]

        # Average the word vectors to get sentence vector
        sentence_vector = np.mean(word_vectors, axis=0)
        
        if sentence_vector is not None:
            # Extract emotions using the emotion extractor
            emotions = get_emo(word2vec_model, loaded_model, sentence_vector)
            result_label.config(text=f"Top Emotions: {emotions}")
        else:
            messagebox.showerror("Error", "No word vectors found in the sentence.")
    else:
        messagebox.showerror("Error", "Please enter your status.")

window = tk.Tk()
window.title("Emotion Detection")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

window.geometry(f"{screen_width}x{screen_height}+0+0")

bg_image = Image.open("images/nn.jpg")
bg_photo = ImageTk.PhotoImage(bg_image.resize((screen_width, screen_height), Image.BILINEAR))

background_label = tk.Label(window, image=bg_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

frame = tk.Frame(window, bg="white")
frame.place(relx=0.5, rely=0.5, anchor="center")

status_label = tk.Label(frame, text="How do you feel?", bg="white")
status_label.pack(pady=5)
status_entry = tk.Entry(frame, bg="white")
status_entry.pack(pady=5)

submit_button = tk.Button(frame, text="Submit", command=submit_status, bg="#4CAF50", fg="white")
submit_button.pack(pady=5)

result_label = tk.Label(frame, text="", bg="white")
result_label.pack(pady=5)

window.mainloop()
