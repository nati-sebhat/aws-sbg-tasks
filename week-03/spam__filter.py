import pandas as pd

# Load the dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display the first five messages
print(df.head())

# Check the dataset dimensions
print("Dataset shape:", df.shape)

# Count ham and spam messages
print("\nLabel distribution:")
print(df["label"].value_counts())




# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Check the distribution of labels
print("\nMessage label distribution:")
print(df["label"].value_counts())



# Display the first 10 duplicated rows
duplicates = df[df.duplicated(keep=False)]

print(duplicates.head(10))


# Display duplicate rows alongside their original occurrences
duplicate_mask = df.duplicated(keep=False)

duplicate_rows = df[duplicate_mask].sort_values(
    by=["label", "message"]
)

print(duplicate_rows.head(20))



def remove_duplicates(df):
    """Remove duplicate rows and return the cleaned dataset."""
    df = df.drop_duplicates()
    return df


# Remove duplicate rows
df = remove_duplicates(df)

# Check the cleaned dataset
print("Dataset shape after cleaning:", df.shape)
print("Remaining duplicate rows:", df.duplicated().sum())

print("\nUpdated label distribution:")
print(df["label"].value_counts())




# Separate messages and labels
emails = df["message"]
labels = df["label"]

# Check the results
print("Messages:")
print(emails.head())

print("\nLabels:")
print(labels.head())



from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    emails,
    labels,
    test_size=0.2,
    random_state=42
)

# Display the sizes of each set
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

print("\nTraining labels:", len(y_train))
print("Testing labels:", len(y_test))




from sklearn.feature_extraction.text import TfidfVectorizer

# Create the TF-IDF vectorizer
vec = TfidfVectorizer()

# Learn vocabulary and transform training messages
X_train_tfidf = vec.fit_transform(X_train)

# Transform testing messages using the same vocabulary
X_test_tfidf = vec.transform(X_test)

# Display the resulting shapes
print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)





from sklearn.naive_bayes import MultinomialNB

# Create the Naive Bayes classifier
model = MultinomialNB()

# Train the model using the training data
model.fit(X_train_tfidf, y_train)

print("Spam classifier trained successfully!")







from sklearn.metrics import accuracy_score

# Predict labels for the testing messages
predictions = model.predict(X_test_tfidf)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2%}")





# Five messages designed to test the model's weaknesses
tricky_messages = [
    "Congratulations! Claim your fr33 prize now by calling 08001234567.",
    "You have NOT won any prize. This is only a reminder about our meeting.",
    "Dear customer, kindly verify your account details to receive your reward.",
    "Hey, can you send me the free notes from today's class?",
    "Your free subscription expires tomorrow. Please renew it through the official university portal."
]

# Expected labels for our five messages
expected_labels = [
    "spam",
    "ham",
    "spam",
    "ham",
    "ham"
]

print("Number of test messages:", len(tricky_messages))
print("Number of expected labels:", len(expected_labels))






# Convert the tricky messages into TF-IDF features
tricky_tfidf = vec.transform(tricky_messages)

# Predict whether each message is spam or ham
tricky_predictions = model.predict(tricky_tfidf)

# Display the predictions alongside the expected labels
for i, (message, prediction, expected) in enumerate(
    zip(tricky_messages, tricky_predictions, expected_labels),
    start=1
):
    result = "Correct" if prediction == expected else "Misclassified"

    print(f"\nMessage {i}: {message}")
    print(f"Predicted: {prediction}")
    print(f"Expected:  {expected}")
    print(f"Result:    {result}")







    # Create a DataFrame to organize the experiment results
results_df = pd.DataFrame({
    "Message": tricky_messages,
    "Predicted Label": tricky_predictions,
    "Expected Label": expected_labels
})

# Check whether each prediction matches the expected label
results_df["Result"] = results_df.apply(
    lambda row: "Correct"
    if row["Predicted Label"] == row["Expected Label"]
    else "Misclassified",
    axis=1
)

# Display the results
print(results_df)

# Calculate accuracy on the tricky messages
tricky_accuracy = accuracy_score(expected_labels, tricky_predictions)

print(f"Tricky message accuracy: {tricky_accuracy:.2%}")