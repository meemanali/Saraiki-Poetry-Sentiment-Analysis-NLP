import pandas as pd
import matplotlib.pyplot as plt

file_path = r"F:\Python\Saraiki Poetry NLP\updated_saraiki_poetry_data.xlsx"
# file_path = r"pdated_saraiki_poetry_data.xlsx"

# Load dataset
df = pd.read_excel(file_path)

print("Columns:", df.columns.tolist())

# -----------------------------
# Emotion distribution
# -----------------------------
emotion_counts = df['Emotions'].value_counts()

plt.figure(figsize=(8, 5))
emotion_counts.plot(kind='bar')
plt.title("Emotion Distribution in Saraiki Poetry Dataset")
plt.xlabel("Emotion")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Sentiment label distribution
# -----------------------------
label_counts = df['Label'].value_counts()

plt.figure(figsize=(6, 4))
label_counts.plot(kind='bar')
plt.title("Sentiment Label Distribution")
plt.xlabel("Sentiment (p = Positive, n = Negative)")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# -----------------------------
# Emotion vs Sentiment relationship
# -----------------------------
emotion_label_matrix = pd.crosstab(df['Emotions'], df['Label'])

plt.figure(figsize=(7, 5))
plt.imshow(emotion_label_matrix.values)
plt.colorbar(label="Count")

plt.xticks(
    ticks=range(len(emotion_label_matrix.columns)),
    labels=emotion_label_matrix.columns
)
plt.yticks(
    ticks=range(len(emotion_label_matrix.index)),
    labels=emotion_label_matrix.index
)

plt.xlabel("Sentiment Label")
plt.ylabel("Emotion")
plt.title("Emotion to Sentiment Mapping")

plt.tight_layout()
plt.show()