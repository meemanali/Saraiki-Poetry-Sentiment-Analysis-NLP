import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your Saraiki poetry data from Excel file
data = pd.read_excel("updated_saraiki_poetry_data.xlsx")

# Extract poetry, sentiments, and emotions
poetry = data['Data'].tolist()
sentiments = data['Label'].tolist()
emotions = data['Emotions'].tolist()

# Convert sentiments to numerical values assuming 'p' (positive) and 'n' (negative)
sentiments = [1 if sentiment == 'p' else 0 for sentiment in sentiments]

# Define parameters
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = int(len(poetry) * 0.8)  # 80% for training, 20% for testing

# Tokenization
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(poetry)
word_index = tokenizer.word_index

# Sequencing and Padding
sequences = tokenizer.texts_to_sequences(poetry)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Splitting into training and testing sets
training_padded = np.array(padded[:training_size])
testing_padded = np.array(padded[training_size:])
training_labels = np.array(sentiments[:training_size])
testing_labels = np.array(sentiments[training_size:])

# Model Definition for Sentiment Analysis
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Summary
model.summary()

# Training the Model
num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

# Convert the model to TensorFlow Lite format for mobile deployment
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the converted model to a .tflite file
    with open("saraiki_sentiment_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("Model has been converted to TFLite and saved as saraiki_sentiment_model.tflite")
except Exception as e:
    print("Error converting model to TensorFlow Lite:", e)

# Function to make predictions
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    prediction = model.predict(padded_sequence)
    return prediction[0][0]

# Example usage of the prediction function
example_poetry = "آپ کُوں وی ملݨ تُوں رہ ڳئے ہیں، ہُݨ تاں نکھرݨ دے ݙر وی نئیں"
predicted_sentiment = predict_sentiment(example_poetry)
print(f"Predicted sentiment for the example poetry: {predicted_sentiment}")


