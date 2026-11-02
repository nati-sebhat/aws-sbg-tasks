# SMS Spam Filter: Breaking a Machine Learning Model

## Project Overview

This project builds an SMS spam classifier using TF-IDF and Multinomial Naive Bayes. The goal is to train a model to distinguish between legitimate messages (`ham`) and unwanted messages (`spam`), evaluate its accuracy, and investigate its weaknesses using carefully designed messages.

## Dataset

I used the [SMS Spam Collection Dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

- Total messages: 5,572
- Ham messages: 4,825
- Spam messages: 747

The dataset is not included in this repository. Download it from the link above and place the extracted `SMSSpamCollection` file inside a local `data/` folder.

## Model and Approach

The project uses:

- **Pandas:** Loading and exploring the dataset.
- **Scikit-learn:** Splitting the dataset, converting text into numerical features, training the classifier, and evaluating its performance.
- **TF-IDF Vectorizer:** Converting SMS messages into numerical features.
- **Multinomial Naive Bayes:** Classifying messages as spam or ham.

The dataset was divided into 80% training data and 20% testing data.

## Model Evaluation

The classifier achieved **96.68% accuracy** on the original testing dataset.

## Tricky Message Experiment

I created five messages to investigate how the model handles unusual spelling, negation, misleading context, and businesslike language.

| My Message | Model Said | Should Be | Why it failed / Result |
|---|---|---|---|
| "Congratulations! Claim your fr33 prize now by calling 08001234567." | spam | spam | Correctly classified despite unusual spelling. |
| "You have NOT won any prize. This is only a reminder about our meeting." | ham | ham | Correctly classified a legitimate message containing negation. |
| "Dear customer, kindly verify your account details to receive your reward." | ham | spam | False negative. Its polite, businesslike wording may have made it resemble legitimate messages. |
| "Hey, can you send me the free notes from today's class?" | ham | ham | Correctly recognized the legitimate educational context despite the word "free". |
| "Your free subscription expires tomorrow. Please renew it through the official university portal." | ham | ham | Correctly classified a legitimate subscription reminder. |

### Experiment Results

- Correct predictions: 4 out of 5
- Misclassified messages: 1
- Accuracy on the five tricky messages: **80%**
- False negatives: 1
- False positives: 0

## What I Learned

The Naive Bayes spam classifier achieved 96.68% accuracy on the original test dataset, but its performance dropped to 80% on my five specially designed messages. I learned that even a model with high accuracy can mistake suspicious messages for legitimate ones when their wording resembles normal business communication.

## How to Run the Project

1. Clone this repository.
2. Download the SMS Spam Collection dataset from the UCI link above.
3. Extract the `SMSSpamCollection` file into a local `data/` folder.
4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Open `spam_filter.ipynb` in Jupyter Notebook or VS Code.
6. Select the project's Python environment and run the notebook cells.

## Project Files

- `spam_filter.ipynb`: Notebook containing the code, model training, evaluation, and experiment results.
- `requirements.txt`: Project dependencies.
- `README.md`: Project documentation.
    