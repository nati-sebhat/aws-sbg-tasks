# Mini Semantic Search

## Overview

Mini Semantic Search is a machine learning and NLP project that retrieves relevant customer support questions and answers based on the meaning of a user's query.

Unlike traditional keyword search, semantic search can identify related questions even when they use different wording.

## Run in Google Colab

You can open and run the notebook directly in Google Colab:

[Open Mini Semantic Search in Google Colab](https://colab.research.google.com/drive/13iB0Z0IfN7OZHeNc6DWvY08L1D97paBM?usp=sharing)

Before running the notebook, install the required dependencies and ensure the dataset is accessible in the Colab environment.


## Features

- Loads customer support data using Pandas.
- Cleans and prepares questions and answers.
- Generates text embeddings using Sentence Transformers.
- Stores embeddings and associated data in ChromaDB.
- Retrieves the three most relevant questions and answers for a query.
- Uses reusable Python functions to organize the workflow.

## Technologies Used

- Python
- Pandas
- NumPy
- Sentence Transformers
- ChromaDB
- Jupyter Notebook

## Project Structure

```text
week-02/
├── data/
├── notebooks/
│   ├── semantic_search.ipynb
│   └── semantic_search.py
├── README.md
└── requirements.txt
```

The `data/` directory contains the dataset used by the notebook. The ChromaDB database is generated locally when the notebook runs.

## Setup

1. Clone the repository.
2. Navigate to the project directory.
3. Create and activate a Python virtual environment.
4. Install the required dependencies:

```bash
pip install -r week-02/requirements.txt
```

5. Open `notebooks/semantic_search.ipynb` in Jupyter Notebook or VS Code.
6. Run the notebook cells in order.



## Example Queries

Try these queries in the notebook:

```python
search_faqs(
    query="How can I cancel my order?",
    collection=collection,
    model=model
)
```

```python
search_faqs(
    query="Where is my package?",
    collection=collection,
    model=model
)
```

```python
search_faqs(
    query="I was charged money twice for one purchase",
    collection=collection,
    model=model
)
```

## How It Works

1. Load the customer support dataset.
2. Select and clean the question-and-answer pairs.
3. Convert the questions into numerical embeddings.
4. Store the questions, answers, and embeddings in ChromaDB.
5. Convert a user's query into an embedding.
6. Retrieve and display the three closest matching questions and their answers.

## Limitations

Search quality depends on the dataset and the embedding model. Retrieved results may be related to a query without answering it exactly. The system retrieves existing answers and does not generate new answers.


## Dataset Attribution

This project uses the **Bitext Customer Support Training Dataset**, published by Bitext.

Dataset source: [Bitext on Hugging Face](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

The dataset provides customer support questions and responses used to develop and test this semantic search system. Credit goes to Bitext for making the dataset available.


## Future Improvements

- Improve retrieval accuracy.
- Add a similarity threshold to filter weak matches.
- Build a simple user interface.
- Support additional customer support datasets.
