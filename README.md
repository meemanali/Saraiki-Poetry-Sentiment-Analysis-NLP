# Saraiki Poetry Sentiment Analysis (NLP)

## Overview

This project explores sentiment analysis of Saraiki poetry, a low-resource regional language, using classical NLP preprocessing and a lightweight deep learning model.

The focus is on:
* Manual data collection and labeling
* Emotion normalization and sentiment abstraction
* End-to-end NLP pipeline using Python and TensorFlow
* Preparing the model for potential mobile (Android) deployment via TensorFlow Lite

## Dataset

* Language: Saraiki
* Domain: Poetry
* Total samples: 958 (full dataset kept private)
* Public release: Partial sample only

<div align="center">
  <img src="https://github.com/meemanali/Saraiki-Poetry-Sentiment-Analysis-NLP/blob/main/Analysis/Collected%20Data.PNG" alt="Emotion distribution" width="420" title="Emotion distribution">
</div>

## Annotation Strategy

Each poem was manually labeled with an emotion:
* happy, sad, surprise, fear, anger, disgust

Emotions were mapped to binary sentiment:
* Positive (p): happy, surprise
* Negative (n): sad, fear, anger, disgust

Only a subset of the dataset is uploaded to illustrate:

* Data format
* Labeling approach
* Reproducibility
* The complete dataset is retained for possible future academic use.

## Data Cleaning & Analysis

Before training, the dataset was cleaned and analyzed to ensure consistency:
* Normalized inconsistent emotion labels (case differences, spelling variants)
* Verified emotion and sentiment distributions
* Analyzed emotion-to-sentiment relationships

## Dataset Analysis (Visual)

The following visualizations summarize the dataset structure:

* Emotion distribution:

<div align="center">
  <img src="https://github.com/meemanali/Saraiki-Poetry-Sentiment-Analysis-NLP/blob/main/Analysis/Emotion%20Distribution.png" alt="Emotion distribution" width="420" title="Emotion distribution">
</div>

* Sentiment (positive / negative) distribution:

<div align="center">
<img src="https://github.com/meemanali/Saraiki-Poetry-Sentiment-Analysis-NLP/blob/main/Analysis/Sentiments%20Labeling.png" alt="Sentiment distribution" width="420" title="Sentiment distribution">
</div>

* Emotion → sentiment mapping:

<div align="center">
<img src="https://github.com/meemanali/Saraiki-Poetry-Sentiment-Analysis-NLP/blob/main/Analysis/Emotion%20Sentiments%20Mapping.png" alt="Sentiment mapping" width="420" title="Sentiment mapping">
</div>

## Methodology

### Preprocessing

* Data loaded from Excel files using Pandas
* Text tokenized using Keras Tokenizer
* Sequences padded/truncated to fixed length
* Out-of-vocabulary handling enabled

### Model

A lightweight neural network was used due to dataset size:
* Embedding layer
* Global Average Pooling
* Dense (ReLU)
* Output layer (Sigmoid)

The model performs binary sentiment classification, not multi-class emotion prediction.

### Training

* Train / test split: 80% / 20%
* Loss function: Binary Crossentropy
* Optimizer: Adam
* Epochs: 30

The model learns general sentiment patterns from poetic text rather than fine-grained emotional nuance.

## Mobile Deployment

After training, the model is converted to TensorFlow Lite (TFLite).
This enables:
* On-device inference
* Low latency predictions
* Future integration with Android applications

## Limitations

* Small dataset size
* Possible class imbalance
* Binary sentiment only
* Simple model architecture

Results should be interpreted as exploratory and baseline-level, not production-ready.

## Technologies Used

* Python
* Pandas
* Matplotlib
* NLTK
* TensorFlow / Keras
* TensorFlow Lite

## Note
This project emphasizes data preparation, language challenges, and applied ML workflow in a low-resource setting rather than optimized model performance.
