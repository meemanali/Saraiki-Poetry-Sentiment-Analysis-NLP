import pandas as pd

# Read the Excel file
train_data = pd.read_excel("updated_file.xlsx", usecols=['Data','Emotions', 'Label' ])

# Define the mapping dictionary
emotion_mapping = {
    'sad': 'sad', 'Sad': 'sad',
    'happy': 'happy', 'Happy': 'happy',
    'surprise': 'surprise', 'Surpise': 'surprise', 'Surprise': 'surprise',
    'fear': 'fear', 'Fear': 'fear',
    'disgust': 'disgust', 'Disgust': 'disgust',
    'anger': 'anger', 'Angry': 'anger', 'anger': 'anger'
}

# Standardize the emotion column
train_data['Emotions'] = train_data['Emotions'].map(emotion_mapping)

# Verify the changes
print(train_data['Emotions'].value_counts())

# Save the cleaned data if needed
train_data.to_excel("Cleaned_Akbar_Urdu_file.xlsx", index=False)
