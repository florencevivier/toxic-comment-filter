# Toxic Comment Multi-Label Classification

## Project Overview

This project aims to build a deep learning model capable of detecting multiple types of toxic content in online comments.

The task is a multi-label classification problem, where each comment can belong to one or more of the following categories:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The dataset is highly imbalanced, with rare classes such as "threat" and "identity_hate" representing less than 1% of the data. Handling imbalance effectively is therefore a central challenge.

The project includes:
- Exploratory Data Analysis (EDA)
- Text preprocessing and cleaning
- Multi-label stratified splitting
- Data augmentation
- Oversampling with MLSMOTE
- Deep Learning modeling (Feedforward + LSTM)
- Custom weighted loss function
- Threshold tuning
- Model comparison and evaluation


## Business Objective

The goal is to automatically detect and classify toxic comments in online platforms.

From a content moderation perspective:

- False Negatives (toxic comments not detected) are highly problematic.
- False Positives (non-toxic comments flagged) are undesirable but less critical.

Therefore, the objective is to maximize macro F1-score while significantly improving recall on minority classes.


## Dataset

The dataset comes from the Kaggle Toxic Comment Classification Challenge.

Each observation contains:
- comment_text
- Six binary toxicity labels

Characteristics:
- Multi-label classification problem
- Strong class imbalance (~90% non-toxic comments)
- Rare labels under 1%
- Comments up to 5000 characters


## Exploratory Data Analysis

Key findings:

- Severe class imbalance across labels.
- No missing or empty comments detected.
- Some duplicated rows removed.
- Comment length alone is not strongly correlated with toxicity.
- Severe toxic comments tend to have higher average length.
- WordCloud analysis highlights discriminative vocabulary per label.

Analysis performed:
- Label distribution percentages
- Length distribution (KDE plots)
- Modal and mean length comparison
- WordCloud per toxicity category


## Preprocessing

Text cleaning pipeline:
- Lowercasing
- Punctuation removal
- Lemmatization (SpaCy)
- Stopword removal (NLTK)
- Digit removal

Tokenization:
- Vocabulary limited to 10,000 most frequent words
- Padding length set to 250 tokens (98th percentile coverage)

Data splitting:
- Multi-label stratified train/validation/test split (70/15/15)

Imbalance handling:
- Data augmentation using additional labeled toxic comments
- Custom MLSMOTE oversampling
- Weighted binary crossentropy loss


## Models Implemented

### 1. Basic Neural Network

Embedding → GlobalAveragePooling → Dense → Dropout → Output

Result:
- Macro F1 ≈ 25%
- Poor detection of minority classes


### 2. Bidirectional LSTM Model

Embedding → BiLSTM → LSTM → Dense → Dropout → Output

Result:
- Macro F1 ≈ 38%
- Improved performance on frequent labels
- Rare classes still poorly detected


### 3. LSTM with Weighted Binary Crossentropy (Final Model)

Same architecture as Model 2, with custom weighted loss:

Weight per label:
w = N_negative / N_positive

Result:
- Macro F1 ≈ 57%
- Significant improvement on rare labels:
  - severe_toxic ≈ 46%
  - identity_hate ≈ 47%
  - threat ≈ 27%

This model achieves the best balance between precision and recall across all classes.


## Threshold Optimization

Threshold values between 0.2 and 0.8 were tested.

Best macro F1 obtained around:
- 0.45 – 0.50

Default threshold of 0.5 was retained.


## Final Model Performance (Test Set)

| Metric | Value |
|--------|--------|
| Macro F1 | ~0.57 |
| Toxic F1 | ~0.76 |
| Severe Toxic F1 | ~0.46 |
| Identity Hate F1 | ~0.47 |
| Threat F1 | ~0.27 |

The weighted LSTM model substantially improves minority class detection compared to baseline models.


## Key Learning Points

- Multi-label classification handling
- Severe class imbalance mitigation
- Cost-sensitive learning (weighted loss)
- Recurrent neural networks for NLP
- Threshold tuning impact on F1-score
- Evaluation beyond accuracy
- Practical trade-offs between precision and recall


## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* NLTK
* SpaCy
* Scikit-learn
* TensorFlow / Keras
* Iterative-Stratification


