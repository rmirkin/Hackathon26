"""
Embedding & Similarity Pipeline
--------------------------------
Generates SBERT embeddings for short text messages and computes
pairwise cosine similarity to surface potential bot/AI-generated content.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --- Configuration ---

MODEL_NAME = "all-MiniLM-L6-v2"

SAMPLE_MESSAGES = [
    "Hey, how's it going?",
    "Hello, how are you doing today?",
    "Buy now and get 50% off on all products!",
    "Limited time offer: 50% discount on everything!",
    "I just finished reading a great book.",
    "The weather is nice today, isn't it?",
    "Click here to claim your free prize immediately!",
    "I enjoyed the movie we watched last night.",
]


# --- Core Functions ---


def load_model(model_name: str) -> SentenceTransformer:
    """Load the sentence-transformer model."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(model: SentenceTransformer, messages: list[str]) -> np.ndarray:
    """Generate embeddings for a list of messages."""
    print(f"Generating embeddings for {len(messages)} messages...")
    embeddings = model.encode(messages, convert_to_numpy=True)
    return embeddings


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between all embeddings."""
    return cosine_similarity(embeddings)


def print_similarity_matrix(sim_matrix: np.ndarray, messages: list[str]) -> None:
    """Print the similarity matrix in a readable format."""
    print("\n=== Cosine Similarity Matrix ===\n")

    # Header row with truncated message labels
    max_label = 20
    labels = [msg[:max_label].ljust(max_label) for msg in messages]

    # Print column indices
    header = " " * (max_label + 2) + "  ".join(f"[{i:>2}]" for i in range(len(messages)))
    print(header)

    # Print each row
    for i, row in enumerate(sim_matrix):
        row_str = "  ".join(f"{val:.2f}" for val in row)
        print(f"{labels[i]}  {row_str}")

    print()


def find_most_similar(sim_matrix: np.ndarray, messages: list[str]) -> None:
    """For each message, find and print the most similar other message."""
    print("=== Most Similar Pairs ===\n")

    for i, msg in enumerate(messages):
        # Mask self-similarity by setting diagonal to -1
        similarities = sim_matrix[i].copy()
        similarities[i] = -1.0

        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        print(f'[{i}] "{msg}"')
        print(f'     -> [{best_idx}] "{messages[best_idx]}" (similarity: {best_score:.4f})')
        print()


# --- Main ---


def main() -> None:
    """Run the full embedding and similarity pipeline."""
    model = load_model(MODEL_NAME)
    embeddings = generate_embeddings(model, SAMPLE_MESSAGES)

    print(f"Embedding shape: {embeddings.shape}")

    sim_matrix = compute_similarity_matrix(embeddings)

    print_similarity_matrix(sim_matrix, SAMPLE_MESSAGES)
    find_most_similar(sim_matrix, SAMPLE_MESSAGES)


if __name__ == "__main__":
    main()
