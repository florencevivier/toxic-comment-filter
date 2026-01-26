# Toxic Comment Filter - RNN-based

## Project Overview
This project implements an advanced deep learning system for automatic moderation of toxic comments in online communities. The system is designed for TechTalk, a technology forum, to help moderators filter offensive, threatening, obscene, or hateful comments in real-time.

## Problem Statement
Manual moderation is inefficient due to the high volume of user comments. Traditional algorithms often fail to capture the complexity and variety of toxic language. The goal of this project is to automate moderation using a recurrent neural network (RNN) to classify comments into multiple toxicity categories.

## Use Case
The community manager, Mario Rossi, cannot manually moderate all comments due to the platform's popularity. The automated system will flag toxic comments quickly, maintaining a safe and inclusive environment for users.

## Dataset
- Source: [Toxic Comments Dataset](https://proai-datasets.s3.eu-west-3.amazonaws.com/Filter_Toxic_Comments_dataset.csv)
- Size: 160,000 comments
- Multi-label categories (each comment can have zero or more labels):
  1. Toxic
  2. Severely Toxic
  3. Obscene
  4. Threat
  5. Insult
  6. Identity Hate

## Model Architecture
- Recurrent neural network (LSTM or GRU)
- Handles sequential nature of text comments
- Multi-label classification output (6 elements per comment)
- Sigmoid activation for each label

## How It Works
1. **Preprocessing:** Tokenization of comments.
2. **Training:** RNN trained to predict multiple toxicity labels.
3. **Inference:** Each comment is processed to output a vector of 6 binary values indicating toxic categories.

## Evaluation
- Metrics: Accuracy, F1-score
- Multi-label classification performance

## Benefits
- **Automation:** Reduces manual moderation workload
- **Efficiency:** Captures context and nuances in text
- **Scalability:** Handles increasing volumes of user comments
- **Integration:** Can be deployed directly in TechTalk’s commenting system



