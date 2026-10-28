from pathlib import Path
import pandas as pd
import numpy as np

from pathlib import Path

# Find the dataset relative to this Python file
data_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)





print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()




# Count duplicate rows
duplicate_count = df.duplicated().sum()

# Count missing values in each column
missing_values = df.isna().sum()

# Count empty or whitespace-only strings in each column
blank_values = df.apply(
    lambda column: column.str.strip().eq("").sum()
)

print("Duplicate rows:", duplicate_count)

print("\nMissing values per column:")
print(missing_values)

print("\nBlank strings per column:")
print(blank_values)




def prepare_data(df):
    # Select the columns needed for semantic search
    faq_df = df[["instruction", "response"]].copy()

    # Rename the columns
    faq_df = faq_df.rename(
        columns={
            "instruction": "question",
            "response": "answer"
        }
    )

    return faq_df


faq_df = prepare_data(df)

print("Prepared dataset shape:", faq_df.shape)
print(faq_df.head().to_string())

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove unnecessary whitespace
    text = " ".join(text.split())

    return text


# Clean questions and answers
faq_df["question"] = faq_df["question"].apply(clean_text)
faq_df["answer"] = faq_df["answer"].apply(clean_text)

print(faq_df.head().to_string())


def clean_dataset(df):
    original_count = len(df)

    # Remove rows with missing questions or answers
    df = df.dropna(subset=["question", "answer"]).copy()

    # Remove rows where either field is empty
    df = df[
        df["question"].str.strip().ne("")
        & df["answer"].str.strip().ne("")
    ].copy()

    # Remove exact duplicate question-answer pairs
    df = df.drop_duplicates(subset=["question", "answer"])

    removed_count = original_count - len(df)

    print("Original records:", original_count)
    print("Records removed:", removed_count)
    print("Remaining records:", len(df))

    return df.reset_index(drop=True)


faq_df = clean_dataset(faq_df)



from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")




def create_embeddings(texts):
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


# Test the function with three example questions
sample_questions = [
    "How can I cancel my order?",
    "I want to return a product.",
    "Where is my package?"
]

sample_embeddings = create_embeddings(sample_questions)

print("Embedding shape:", sample_embeddings.shape)




# Generate embeddings for all customer support questions
questions = faq_df["question"].tolist()

question_embeddings = create_embeddings(questions)

print("Number of questions:", len(questions))
print("Embeddings shape:", question_embeddings.shape)




import chromadb

# Create a ChromaDB client that stores data locally
client = chromadb.PersistentClient(path="../chroma_db")

# Create or retrieve our collection
collection = client.get_or_create_collection(
    name="customer_support_faqs",
    metadata={"hnsw:space": "cosine"}
)

print("ChromaDB collection is ready!")





def store_documents(collection, faq_df, embeddings, batch_size=500):
    total_records = len(faq_df)

    for start in range(0, total_records, batch_size):
        end = min(start + batch_size, total_records)

        # Prepare one batch of records
        batch_df = faq_df.iloc[start:end]

        collection.upsert(
            ids=[str(i) for i in range(start, end)],
            documents=batch_df["question"].tolist(),
            metadatas=[
                {"answer": answer}
                for answer in batch_df["answer"].tolist()
            ],
            embeddings=embeddings[start:end].tolist()
        )

    print(f"Stored {total_records} records in ChromaDB.")


store_documents(collection, faq_df, question_embeddings)




def search_faqs(query, collection, model, top_k=3):
    # Convert the user's query into an embedding
    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    # Find the most relevant stored questions
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # Display the results
    for i in range(len(results["ids"][0])):
        question = results["documents"][0][i]
        answer = results["metadatas"][0][i]["answer"]
        distance = results["distances"][0][i]

        print(f"\nResult {i + 1}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Cosine distance: {distance:.4f}")
        print("-" * 50)




        print("Starting search...")

search_faqs(
    query="Where is my package?",
    collection=collection,
    model=model
)

print("Search finished.")