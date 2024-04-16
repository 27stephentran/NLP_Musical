import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

# Load data from train.json file
with open("data/train.json", "r") as f:
    data = json.load(f)

# Extract features (vectors and emotion probabilities) and labels
X = np.array([entry["vector"] for entry in data])
emotions = np.array([entry["emotions"] for entry in data])

# Convert emotions to integer labels
label_encoder = LabelEncoder()
emotions_array = np.array([list(emotion.values()) for emotion in emotions])

# Now apply argmax to find the index of the maximum value along axis 1
# This will return an array of indices indicating the most probable emotion for each sample
emotion_indices = np.argmax(emotions_array, axis=1)

# Now encode the emotion indices using LabelEncoder
y_encoded = label_encoder.fit_transform(emotion_indices)

# Define the number of classes
num_classes = len(np.unique(y_encoded))

# Convert integer labels to one-hot encoded vectors
from keras.utils import to_categorical
y_one_hot = to_categorical(y_encoded, num_classes=num_classes)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Reshape data for Conv1D layer
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# Define the CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

import matplotlib.pyplot as plt

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
