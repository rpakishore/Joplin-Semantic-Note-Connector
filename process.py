# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "openai",
#     "tiktoken" # Added tiktoken explicitly as it's used
# ]
# ///

import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
from openai import OpenAI
import tiktoken
import tomllib
import json
import hashlib

# --- Configuration ---
CACHE_FILEPATH = "embeddings_cache.json"
RESULTS_MARKDOWN_FILEPATH = "relevance_results.md"  # Output file for results
# --- End Configuration ---

# --- Global OpenAI Client Initialization ---
try:
    with open("config.toml", "rb") as f:
        _data = tomllib.load(f)["llm"]
        OPENAI_MODEL_NAME: str = _data["model"]
        OPENAI_MODEL_CONTEXT: int = _data["model_context"]
        client = OpenAI(
            base_url=_data["url"],
            api_key=_data["key"],
        )
        del _data
except FileNotFoundError:
    print("Error: config.toml not found. Please ensure the configuration file exists.")
    exit(1)
except KeyError:
    print(
        "Error: 'llm' section or its keys ('model', 'url', 'key') not found in config.toml."
    )
    exit(1)
# --- End Global OpenAI Client Initialization ---


# --- Caching Helper Functions ---
def calculate_content_hash(content: str) -> str:
    """
    Calculates SHA256 hash of the content.
    The content should be the exact string that would be sent to the embedding API.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_embeddings_cache(filepath: str = CACHE_FILEPATH) -> dict:
    """
    Loads embeddings cache from a JSON file.
    Keys are content hashes, values are the embedding lists.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            cache = json.load(f)
            print(f"Successfully loaded embeddings cache from {filepath}")
            return cache
    except FileNotFoundError:
        print(f"Cache file {filepath} not found. Starting with an empty cache.")
        return {}
    except json.JSONDecodeError:
        print(
            f"Warning: Cache file {filepath} is corrupted or not valid JSON. Starting with an empty cache."
        )
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while loading the cache: {e}")
        return {}


def save_embeddings_cache(cache_data: dict, filepath: str = CACHE_FILEPATH):
    """Saves embeddings cache to a JSON file."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=4)
        print(f"Successfully saved embeddings cache to {filepath}")
    except IOError as e:
        print(f"Error saving cache to {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the cache: {e}")


# --- End Caching Helper Functions ---


def load_markdown_documents(directory_path: str) -> list:
    """
    Loads all markdown documents from the specified directory.
    Returns a list of tuples (filepath, content).
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return []

    p_directory = Path(directory_path)
    md_files = list(p_directory.glob("**/*.md"))

    if not md_files:
        print(f"No .md files found in '{directory_path}'.")
        return []

    documents = []
    for file_path in md_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append((str(file_path), content))
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    return documents


def generate_ollama_embeddings(
    docs_content: list, embeddings_cache: dict
) -> tuple[np.ndarray | None, bool]:
    """
    Generates embeddings for a list of document contents using the configured OpenAI-compatible API.
    Uses a cache to avoid re-generating embeddings for identical content.

    Args:
        docs_content (list): A list of strings, where each string is the content of a document.
        embeddings_cache (dict): The loaded cache of embeddings.

    Returns:
        tuple:
            - numpy.ndarray: A 2D numpy array of embeddings. None if generation fails critically.
            - bool: True if the cache was modified (new embeddings were added), False otherwise.
    """
    embeddings_list = []
    cache_updated = False
    tokenizer = tiktoken.encoding_for_model(OPENAI_MODEL_NAME)

    num_documents = len(docs_content)
    for i, original_content in enumerate(docs_content):
        print(f"  Processing document {i+1}/{num_documents} for embedding...", end="\r")

        content_for_api = original_content
        if not original_content.strip():
            print(
                f"\nSkipping empty content for document {i+1}/{num_documents}. Using a single space for embedding."
            )
            content_for_api = " "

        tokens = tokenizer.encode(content_for_api)
        if len(tokens) > OPENAI_MODEL_CONTEXT:
            print(
                f"\nContent for document {i+1} is too long ({len(tokens)} tokens). Truncating to {OPENAI_MODEL_CONTEXT} tokens."
            )
            content_for_api = tokenizer.decode(tokens[:OPENAI_MODEL_CONTEXT])

        content_hash = calculate_content_hash(content_for_api)

        if content_hash in embeddings_cache:
            embedding = embeddings_cache[content_hash]
        else:
            try:
                response = client.embeddings.create(
                    model=OPENAI_MODEL_NAME,
                    input=content_for_api,
                    encoding_format="float",
                )
                embedding = response.data[0].embedding
                embeddings_cache[content_hash] = embedding
                cache_updated = True
            except Exception as e:
                print(
                    f"\nError generating embedding for document {i+1}/{num_documents}: {e}"
                )
                print("Critical error during embedding generation. Aborting.")
                return None, cache_updated  # Signal critical failure

        embeddings_list.append(embedding)

    print(f"\nEmbeddings processed for {len(embeddings_list)} documents.              ")
    if not embeddings_list:
        return None, cache_updated
    return np.array(embeddings_list, dtype=np.float32), cache_updated


def find_most_relevant_documents(
    doc_embeddings: np.ndarray,
    doc_filepaths: list,
    current_doc_index: int,
    top_n: int = 5,
) -> list:
    """
    Finds the most relevant documents for a given document based on cosine similarity.
    """
    if (
        doc_embeddings is None
        or doc_embeddings.ndim != 2
        or doc_embeddings.shape[0] <= 1
    ):
        return []
    if current_doc_index >= len(doc_embeddings) or current_doc_index >= len(
        doc_filepaths
    ):
        print(
            f"Warning: current_doc_index {current_doc_index} is out of bounds for embeddings/filepaths."
        )
        return []

    current_embedding = doc_embeddings[current_doc_index].reshape(1, -1)
    similarities = cosine_similarity(current_embedding, doc_embeddings)[0]

    similarity_scores = []
    for i, score in enumerate(similarities):
        if i == current_doc_index:
            continue
        if i >= len(doc_filepaths):
            print(
                f"Warning: Index {i} out of bounds for doc_filepaths during similarity scoring."
            )
            continue
        similarity_scores.append((doc_filepaths[i], float(score), i))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return [(filepath, score) for filepath, score, _ in similarity_scores[:top_n]]


def save_results_to_markdown(
    markdown_content: str, filepath: str = RESULTS_MARKDOWN_FILEPATH
):
    """Saves the generated Markdown content to a file."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Successfully saved relevance results to {filepath}")
    except IOError as e:
        print(f"Error saving results to Markdown file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the results Markdown: {e}")


def main():
    print("Starting document relevance analysis...")
    markdown_output_lines = []

    # Load existing embeddings cache
    embeddings_cache = load_embeddings_cache()

    notes_directory = input("Enter the path to your Joplin Markdown notes directory: ")
    if not notes_directory.strip():
        print("No directory provided. Exiting.")
        return

    documents_data = load_markdown_documents(notes_directory)
    if not documents_data:
        print("No Markdown documents were loaded. Exiting.")
        return

    docs_content_to_embed = []
    doc_filepaths_for_embedding = []

    for filepath, content in documents_data:
        if content.strip():
            docs_content_to_embed.append(content)
            doc_filepaths_for_embedding.append(filepath)
        else:
            print(f"Skipping '{Path(filepath).name}' as it has no substantial content.")

    if not docs_content_to_embed:
        print("No documents with content found to process for embeddings. Exiting.")
        return

    print(
        f"\nFound {len(doc_filepaths_for_embedding)} documents with content to process for embeddings."
    )

    print("Generating embeddings (this may take a while for new documents)...")
    doc_embeddings, cache_was_updated = generate_ollama_embeddings(
        docs_content_to_embed, embeddings_cache
    )

    if doc_embeddings is None:
        print("Could not generate embeddings. Exiting.")
        if cache_was_updated:
            save_embeddings_cache(embeddings_cache)
        return

    if len(doc_embeddings) != len(doc_filepaths_for_embedding):
        print(
            "Mismatch between the number of embeddings generated and the number of documents. Exiting."
        )
        if cache_was_updated:
            save_embeddings_cache(embeddings_cache)
        return

    if cache_was_updated:
        save_embeddings_cache(embeddings_cache)

    try:
        top_n_input = input(
            "How many relevant documents do you want to find for each note? (e.g., 5, default is 5): "
        )
        if not top_n_input.strip():
            top_n_relevant = 5
            print("No input provided for N. Defaulting to 5.")
        else:
            top_n_relevant = int(top_n_input)
        if top_n_relevant <= 0:
            print("Number must be positive. Defaulting to 5.")
            top_n_relevant = 5
    except ValueError:
        print("Invalid input for N. Defaulting to 5 relevant documents.")
        top_n_relevant = 5

    # --- Format Results for Markdown ---
    markdown_output_lines.append("# Document Relevance Analysis Results\n")
    markdown_output_lines.append(f"Source Directory: `{notes_directory}`\n")
    markdown_output_lines.append(
        f"Number of documents processed for embeddings: {len(doc_filepaths_for_embedding)}\n"
    )
    markdown_output_lines.append(
        f"Top N relevant documents requested: {top_n_relevant}\n"
    )
    markdown_output_lines.append("---\n")

    num_docs_with_embeddings = len(doc_filepaths_for_embedding)

    if num_docs_with_embeddings <= 1:
        markdown_output_lines.append(
            "Only one or zero documents with content found. Cannot find relevant documents.\n"
        )
        print(
            "Only one or zero documents with content found. Cannot find relevant documents."
        )
    else:
        print("\nGenerating relevance report...")
        for i, current_filepath_str in enumerate(doc_filepaths_for_embedding):
            current_filepath = Path(current_filepath_str)
            markdown_output_lines.append(f"## For document: {current_filepath.name}\n")
            markdown_output_lines.append(f"- Full Path: `{current_filepath_str}`\n")

            effective_top_n = min(top_n_relevant, num_docs_with_embeddings - 1)

            if effective_top_n <= 0:
                markdown_output_lines.append(
                    "  - No other documents to compare against.\n\n"
                )
                continue

            relevant_docs = find_most_relevant_documents(
                doc_embeddings, doc_filepaths_for_embedding, i, top_n=effective_top_n
            )

            if relevant_docs and relevant_docs[0][1] > 0.6:
                markdown_output_lines.append("  - **Most relevant documents:**\n")
                for idx, (filepath_str, score) in enumerate(relevant_docs):
                    if score < 0.6:
                        continue
                    relevant_filepath = Path(filepath_str)  # Convert to Path
                    markdown_output_lines.append(
                        f"    {idx+1}. **{relevant_filepath.name}** (Similarity: {score:.4f})\n"
                    )
                    markdown_output_lines.append(f"       - Path: `{filepath_str}`\n")
            else:
                markdown_output_lines.append(
                    "  - No relevant documents found for this document.\n"
                )
            markdown_output_lines.append("\n")

    # --- Add Next Steps to Markdown ---
    markdown_output_lines.append("---\n")
    markdown_output_lines.append("## Next Steps\n")
    markdown_output_lines.append("1. Review the relevance scores in this report.\n")
    markdown_output_lines.append(
        "2. If using with Joplin, you might want to map these file paths to Joplin Note IDs "
        "(e.g., `:/NOTE_ID_HERE`) for creating internal links.\n"
    )
    markdown_output_lines.append(
        "   This script identifies relevant *file paths* based on their content similarity.\n"
    )

    # --- Save Markdown File ---
    final_markdown_content = "".join(markdown_output_lines)
    save_results_to_markdown(final_markdown_content)

    print("\n--- End of Console Output ---")
    print(f"Detailed results have been saved to: {RESULTS_MARKDOWN_FILEPATH}")


if __name__ == "__main__":
    main()
