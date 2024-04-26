import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower
from transformers import pipeline
import re
import json


spark = SparkSession.builder \
    .appName("Emotion Detection") \
    .getOrCreate()

emotion_detector = pipeline(model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)


def preprocess_text_safe(text):
    if text is None:
        return ""
    if isinstance(text, str):
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        return text.strip()
    else:
        return ""

def detect_emotion(text):
    try:
       
        return emotion_detector(text, truncation=True)
    except Exception as e:
        print(f"Error during emotion detection: {e}")
        return None

# Function to calculate emotions with Spark DataFrame
def calculate_emotion(filename):
    emotion_level_list = []
    try: 
        with open(filename, 'r') as file:
            data = json.load(file)
        if "utterance" in data:
            for utterance in data["utterance"]:
                clean_text = preprocess_text_safe(utterance)
                detected_emotions = detect_emotion(clean_text)
                emotion_level = {}
                for entry in detected_emotions[0]:
                    emotion_level[entry['label']] =  entry['score']
                emotion_level_list.append(emotion_level)
            return emotion_level_list
    except FileNotFoundError:
        print(f'Error: File not found - {filename}')
    except json.JSONDecodeError:
        print(f'Error: Invalid JSON format in file - {filename}')
    except Exception as e:
        print(f'Error: {str(e)}')

# Function to write data back to JSON
def write_data(filename, emotion_data):
    with open(filename, 'r') as json_file:
        existing_data = json.load(json_file)

    existing_data["emotion_level"] = emotion_data

    with open(filename, 'w') as json_file:
        json.dump(existing_data, json_file, indent=2)

def process_file():
    filename = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON files", "*.json")])
    if filename:
        list_e = calculate_emotion(filename)
        if list_e:
            write_data(filename, list_e)
            messagebox.showinfo("Success", "Emotion data successfully calculated and written to the file.")
        else:
            messagebox.showerror("Error", "Failed to calculate emotion data.")
