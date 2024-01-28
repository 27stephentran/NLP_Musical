from transformers import pipeline
from clean_context import *
import json
emotion_detector = pipeline(model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None) 

def detect_emotion(text):
    try:
        # Pass the raw text directly to the emotion detector
        return emotion_detector(text, truncation=True)
    except Exception as e:
        print(f"Error during emotion detection: {e}")
        return None
    
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
                print(emotion_level)
                emotion_level_list.append(emotion_level)
            return emotion_level_list
    except FileNotFoundError:
        print(f'Error: File not found - {filename}')
    except json.JSONDecodeError:
        print(f'Error: Invalid JSON format in file - {filename}')
    except Exception as e:
        print(f'Error: {str(e)}') 