# Sentiment-Classification-with-BERT-and-Hugging-Face
1. Model Overview
Description: This project involves building a Sentiment Classifier for the IMDB dataset. The model utilizes BERT (Bidirectional Encoder Representations from Transformers) from the Transformers library by Hugging Face, implemented with PyTorch.

Components:

Model Architecture: BERT-based model for sequence classification.
Framework: PyTorch and Transformers library.
2. Dataset Description
Dataset: The IMDB dataset consists of 50,000 movie reviews used for sentiment analysis. It is divided into:

Training Set: 25,000 reviews.
Test Set: 25,000 reviews.
Source: The dataset can be downloaded from Kaggle IMDB Dataset:- https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Task: Binary sentiment classification (positive or negative reviews).

3. Dataset Preprocessing
Objective: Convert raw text data into a format suitable for machine learning models.

Steps:

Tokenization: Convert text into tokens using BERT's tokenizer.
Special Tokens: Add special tokens to indicate sentence boundaries and classification.
Padding: Ensure all sequences are of a constant length by adding padding tokens.
Attention Mask: Create an attention mask to differentiate between real tokens and padding tokens.
Tools: The Transformers library provides pre-built tokenizers and models.

4. Creation of Sentiment Classifier using Transfer Learning and BERT
Model Architecture:

Base Model: Pre-trained BERT model (BertModel).
Classifier Layer: Built on top of the BERT base model, including:
Dropout Layer: For regularization.
Fully-Connected Layer: For classification into positive or negative sentiment.
Implementation:

Use the pre-trained BERT model from Hugging Face.
Add a classification head (fully-connected layer) to output sentiment predictions.
5. Training the Model
Optimizer: AdamW (Adam with Weight Decay) from the Transformers library.

Scheduler: Linear learning rate scheduler with warmup steps.

Hyperparameters:

Batch Size: 16
Learning Rate: 2e-5
Number of Epochs: 4 (or as determined based on early stopping)
Training Loop:

Implement training and validation phases.
Utilize gradient clipping and early stopping to improve training stability.
6. Model Evaluation
Evaluation Metrics:

Softmax Activation: Applied to model outputs to obtain probabilities.
Accuracy: Calculate and plot training and validation accuracy.
Classification Report: Includes precision, recall, and F1-score.
Confusion Matrix: To visualize model performance on different classes.
Visualization:

Accuracy Graph: Plot training and validation accuracy over epochs.
Confusion Matrix: Display confusion matrix for a detailed performance analysis.
