# SMS Spam Classifier

## Overview

This project builds a machine learning model that classifies SMS messages as either spam or ham (legitimate messages). It uses Natural Language Processing (NLP) techniques to process text and a Multinomial Naive Bayes classifier to make predictions.

The project also tests the model using five specially designed messages to explore its strengths and weaknesses.

## Google Colab

You can view and run the notebook, including its saved outputs, in Google Colab:

[Open SMS Spam Classifier in Google Colab](https://colab.research.google.com/drive/1P9EwyQy1AGeCuKOFi3tEPSEmeW-GM4ZZ?usp=sharing)

## Dataset

The project uses the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

The original dataset contains 5,572 SMS messages labeled as spam or ham.

### Data Preprocessing

Before training the model, I checked the dataset for missing values and duplicate records.

- Missing values found: 0
- Duplicate rows found: 403
- Duplicate rows removed: 403
- Original dataset size: 5,572 messages
- Dataset size after cleaning: 5,169 messages

After removing duplicates, the cleaned dataset contained 4,516 ham messages and 653 spam messages.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook
- TF-IDF Vectorization
- Multinomial Naive Bayes

## How It Works

1. Load the SMS dataset using Pandas.
2. Check for missing values and duplicate records.
3. Remove duplicate rows.
4. Separate messages from their labels.
5. Split the dataset into training and testing sets.
6. Convert the text messages into numerical features using TF-IDF.
7. Train a Multinomial Naive Bayes classifier.
8. Evaluate the model using test data.
9. Test the model with five specially designed messages.

## Model Results

The cleaned dataset was divided into training and testing sets.

| Measurement | Result |
|---|---:|
| Total messages after cleaning | 5,169 |
| Training samples | 4,135 |
| Testing samples | 1,034 |
| TF-IDF features | 7,585 |
| Model | Multinomial Naive Bayes |
| Test accuracy | 95.84% |

The model achieved **95.84% accuracy** on the test dataset.

## Tricky Message Experiments

I tested the classifier using five messages designed to challenge its predictions. The experiments included unusual spelling, negation, polite wording, misleading context, and legitimate messages containing words commonly associated with spam.

| My message | Model said | Should be | Why it failed |
|---|---|---|---|
| Congratulations! Claim your fr33 prize now by calling 08001234567. | Spam | Spam | No failure. The model correctly classified the message despite the unusual spelling of "free." |
| You have NOT won any prize. This is only a reminder about our meeting. | Ham | Ham | No failure. The model correctly identified the legitimate context despite words associated with spam. |
| Dear customer, kindly verify your account details to receive your reward. | Ham | Spam | False negative. The polite wording may have resembled legitimate communication. |
| Hey, can you send me the free notes from today's class? | Ham | Ham | No failure. The model correctly recognized the educational context despite the word "free." |
| Your free subscription expires tomorrow. Please renew it through the official university portal. | Ham | Ham | No failure. The model correctly classified the legitimate reminder. |

### Experiment Summary

- Correct predictions: 4 out of 5
- Incorrect predictions: 1
- Accuracy on the five tricky messages: 80%
- False negatives: 1
- False positives: 0

The model incorrectly classified the third message as ham even though it was intended to represent spam. Its polite, professional wording may have made the message harder to distinguish from legitimate communication.

These experiments demonstrate that a model can achieve high test accuracy while still making mistakes on individual messages.

## What I Learned

I learned how to preprocess a text dataset, convert messages into TF-IDF features, and train a Naive Bayes classifier for spam detection. I also learned that removing duplicate records can affect model evaluation and that high accuracy does not guarantee correct predictions for every message.

## How to Run the Project

1. Clone the repository.
2. Navigate to the `week-03` directory.
3. Create and activate a Python virtual environment.
4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Download the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
6. Place the extracted `SMSSpamCollection` file inside a local `data/` directory.
7. Open `spam_filter.ipynb` in Jupyter Notebook or VS Code.
8. Run the notebook cells in order.

The dataset is not included in this repository and must be downloaded separately.

## Project Structure

```text
week-03/
├── data/                  # Local dataset, not uploaded to GitHub
├── spam_filter.ipynb      # Notebook containing code and results
├── spam__filter.py        # a python file containing the source code
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── .gitignore              # Files excluded from Git
```