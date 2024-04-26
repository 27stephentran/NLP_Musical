import tkinter as tk
<<<<<<< HEAD
from tkinter import messagebox
from PIL import Image, ImageTk
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array_contains
from pyspark.sql import Row
from keras.models import model_from_json
import gensim.downloader as api
import numpy as np
from processing_context.get_mood import EmotionExtractor
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import rand, expr
import random
import json

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("Emotion Detection") \
    .getOrCreate()

training_data =  spark.read.json("data/train.json")
# Load pre-trained model
=======
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from processing_context.clean_context import preprocess_text_safe
from processing_context.get_mood import EmotionExtractor
import gensim.downloader as api
import numpy as np
from keras.models import model_from_json
import json

>>>>>>> 3e70dfd5ecc30d2b0267202abf7e5ee39f32788c
with open("custom_model/my_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

<<<<<<< HEAD
# Load word2vec model
word2vec_model = api.load("word2vec-google-news-300")

# Define the schema for the JSON data
schema = StructType([
    StructField("name", StringType(), True),
    StructField("artist", StringType(), True),
    StructField("lyric", StringType(), True),
    StructField("genre", StringType(), True),
    StructField("top_emotions", ArrayType(StringType()), True)
])

# Đọc dữ liệu từ tệp JSON vào DataFrame
df = spark.read.format("json").schema(schema).load("data/songs_with_emotions.json")

# Function to filter songs based on emotions
def filter_songs(emotions):
    # Lấy 2 trong 3 top_emotions đã phân tích một cách ngẫu nhiên
    random_emotions = random.sample(emotions, 2)
    
    # Lọc các bài hát có top_emotions được chọn ngẫu nhiên
    songs_with_emotion = df.filter(array_contains(col("top_emotions"), random_emotions[0]) | array_contains(col("top_emotions"), random_emotions[1]))

    # Chỉ hiển thị cột "name" của các bài hát thỏa mãn điều kiện
    filtered_songs = songs_with_emotion.select("name")
    
    # Chọn một bài hát ngẫu nhiên từ DataFrame nếu danh sách bài hát rỗng
    if filtered_songs.count() == 0:
        #print(f"Không có bài hát nào chứa cảm xúc {random_emotions[0]} hoặc {random_emotions[1]}.")#
        return get_all_song_names()
    else:
        # Trả về DataFrame chứa danh sách bài hát thỏa mãn điều kiện
        return filtered_songs

# Hàm phân tích cảm xúc từ câu
=======
word2vec_model = api.load("word2vec-google-news-300")

saved_status = None  

>>>>>>> 3e70dfd5ecc30d2b0267202abf7e5ee39f32788c
def get_emo(word2vec_model, loaded_model, sentence):
    extractor = EmotionExtractor(word2vec_model, loaded_model)
    top_emotions = extractor.extract_top_emotions(sentence)
    return top_emotions

<<<<<<< HEAD
# Hàm hiển thị kết quả trong bảng



def display_results(filtered_songs):
    result_window = tk.Toplevel()
    result_window.title("Danh sách bài hát phù hợp")

    if filtered_songs:
        # Get the names of songs from the DataFrame
        song_names = [row["name"] for row in filtered_songs if "name" in row]
        if song_names:
            # Display a random sample of 20 song names from the filtered songs
            random_song_names = random.sample(song_names, min(len(song_names), 20))
            for song_name in random_song_names:
                song_label = tk.Label(result_window, text=song_name, fg="black")
                song_label.pack(pady=5)
        else:
            # Show all song names from the JSON file
            all_song_names = get_all_song_names()
            if all_song_names:
                # Display a random sample of 20 song names from all songs in the JSON file
                random_song_names = random.sample(all_song_names, min(len(all_song_names), 20))
                for song_name in random_song_names:
                    song_label = tk.Label(result_window, text=song_name, fg="black")
                    song_label.pack(pady=5)
            else:
                messagebox.showinfo("Info", "Không có bài hát nào trong dữ liệu.")
    else:
        messagebox.showinfo("Info", "Không có bài hát nào trong dữ liệu.")

# Hàm trả về danh sách tên bài hát từ tệp JSON
def get_all_song_names():
    song_names = []
    with open("data/songs_with_emotions.json", "r") as json_file:
        data = json.load(json_file)
        for song_key, song_info in data.items():
            if "name" in song_info:
                song_names.append(song_info["name"])
            else:
                # Handle the case where "name" field is missing
                print(f"Warning: 'name' field missing for song with key '{song_key}'")
    return song_names

# Sử dụng hàm để lấy danh sách tên bài hát
song_names = get_all_song_names()
print(song_names)
def submit_status():
    # Get the status from the entry widget
    status = status_entry.get()
    
    # Check if the status is not empty
    if status:
        # Define the schema for the DataFrame
        # schema = StructType([
        #     StructField("sentence", StringType(), True),
        #     StructField("vector", StringType(), True),
        #     StructField("emotions", ArrayType(StringType()), True)
        # ])

        # # Create a new Row for the DataFrame
        # new_row = Row(sentence=status, vector="", emotions=[])

        # # Create a DataFrame from the new Row using the defined schema
        # new_df = spark.createDataFrame([new_row], schema=schema)

        # # Append the new DataFrame to the existing training data DataFrame
        # updated_df = training_data.union(new_df)

        # # Write the updated DataFrame to a JSON file
        # updated_df.write.mode("overwrite").json("data/train.json")

        # Process the sentence to extract emotions and display results
        word_vectors = [word2vec_model[word] for word in status.split() if word in word2vec_model]
=======
def submit_status():
    global saved_status
    status = status_entry.get()
    if status:
        sentence = status
        
        # Vectorize the sentence using the word2vec model
        word_vectors = [word2vec_model[word] for word in sentence.split() if word in word2vec_model]

        # Average the word vectors to get sentence vector
>>>>>>> 3e70dfd5ecc30d2b0267202abf7e5ee39f32788c
        sentence_vector = np.mean(word_vectors, axis=0)
        
        if sentence_vector is not None:
            # Extract emotions using the emotion extractor
            emotions = get_emo(word2vec_model, loaded_model, sentence_vector)
<<<<<<< HEAD
            
            # Update the result label with the top emotions
            result_label.config(text=f"Top Emotions: {emotions}")
            
            # Filter songs based on emotions and display the results
            filtered_songs = filter_songs(emotions)
            display_results(filtered_songs)
            
        else:
            # Show an error message if no word vectors are found
            messagebox.showerror("Error", "No word vectors found in the sentence.")
    else:
        # Show an error message if the status entry is empty
        messagebox.showerror("Error", "Please enter your status.")
=======
            result_label.config(text=f"Top Emotions: {emotions}")
        else:
            messagebox.showerror("Error", "No word vectors found in the sentence.")
    else:
        messagebox.showerror("Error", "Please enter your status.")

>>>>>>> 3e70dfd5ecc30d2b0267202abf7e5ee39f32788c
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
<<<<<<< HEAD

# Đóng SparkSession
spark.stop()
=======
>>>>>>> 3e70dfd5ecc30d2b0267202abf7e5ee39f32788c
