import os
from typing import List
import pandas as pd



def generate_text_embeddings(texts: List[str], embedding_fn: callable,
                             precision: int = 6) -> list:
    """Generate embeddings for a list of texts using the specified model."""

    # Clean up each text by replacing newline characters
    cleaned_texts = [text.replace("\n", " ") for text in texts]

    # Request embeddings for all texts at once from the OpenAI API
    embeddings = embedding_fn(cleaned_texts)

    # Round embedding values to the specified precision
    embeddings = [[round(value, precision) for value in embedding] for embedding in embeddings]
    return embeddings


def embed_dataframe(df: pd.DataFrame, output_file: str, embedding_fn: callable, 
                    show_preview: bool = True) -> None:
    """Embed text data in the provided DataFrame using the specified model and save it to a CSV file."""

    # Generate embeddings for all text data in the DataFrame
    texts = df["text"].tolist()
    embeddings = generate_text_embeddings(texts, embedding_fn)

    # Add embeddings to the DataFrame
    df["embedding"] = embeddings

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the DataFrame with embeddings to a CSV file
    df.to_csv(output_file, index=False)

    # Display a preview of the DataFrame if requested
    if show_preview:
        pd.set_option("display.max_colwidth", 150)
        print(df.head(10))

    print(f"\n\n***** Embeddings saved to '{output_file}'.")