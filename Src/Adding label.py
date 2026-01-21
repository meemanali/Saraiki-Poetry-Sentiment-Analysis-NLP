import pandas as pd

# Read the Excel file
df = pd.read_excel('saraiki_poetry_raw_data.xlsx')

# Define a function to map emotions to labels
def map_emotions_to_label(emotion):
    if emotion in ['happy', 'surprise']:
        return 'p'
    elif emotion in ['sad', 'disgust', 'anger', 'fear']:
        return 'n'
    else:
        return None  # Handle other cases if any

# Apply the function to create the Label column
df['Label'] = df['Emotions'].apply(map_emotions_to_label)

# Write the updated DataFrame back to a new Excel file
df.to_excel('updated_file_3.xlsx', index=False)

print("done")