import csv
from datetime import datetime
from itertools import product
from difflib import SequenceMatcher
import json
import logging
import os
import re

import torch

from aspect_based import AspectSentimentExtractor, pool_embeddings
from config import DISTILBERT_BASE, DEVICE, BATCH_SIZE, OVERLAP_KEYWORDS_FOR_RELEASE_FILTERING
from preprocess import load_aspect_labels
from processing import tokenize

logger = logging.getLogger(__name__)

def load_csv_rows(file_path: str) -> list[dict]:
    """Load CSV rows into a list of dictionaries."""

    with open(file_path, 'r') as file:
        return list(csv.DictReader(file))
    
def extract_keyword_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())

def encode_text(model: AspectSentimentExtractor, rows: list[dict], is_review: bool = False) -> dict[int, dict]:
    """Find embeddings and aspect/sentiment predictions for a list of text rows (from reviews or release notes) using the provided model

    Args:
        model (AspectSentimentExtractor): Model for generating embeddings and aspect/sentiment predictions
        rows (list[dict]): List of dictionaries containing text data to encode
        is_review (bool, optional): If the text provided is from reviews, if so, give sentiments as well. Defaults to False.

    Returns:
        dict[int, dict]: Dictionary mapping row IDs to their content, aspect/sentiment predictions and embeddings
    """

    data = dict()

    with torch.no_grad():
        for i in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[i:i+BATCH_SIZE]

            inputs = tokenize([row["content"] for row in batch_rows], DISTILBERT_BASE)
            input_ids = inputs['input_ids'].to(DEVICE)
            attention_mask = inputs['attention_mask'].to(DEVICE)

            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

            mean_pooled = pool_embeddings(embeddings, attention_mask)

            aspect, sentiment = model.aspect_sentiment_inference(
                input_ids=input_ids,
                attention_masks=attention_mask,
                batch_size=BATCH_SIZE,
            )

            for batch_idx, row in enumerate(batch_rows):
                row['embedding'] = mean_pooled[batch_idx].cpu().numpy()
                row['aspect'] = aspect[batch_idx]
                row['tokens'] = extract_keyword_tokens(row["content"])

                # Only add sentiment if the text is from reviews, not release notes
                if is_review:
                    row['sentiment'] = sentiment[batch_idx]

                data[i + batch_idx] = row

        return data

def filter_pairs(release_note_data: dict[int, dict], review_data: dict[int, dict]) -> list[tuple[dict, dict]]:
    """Filters (release note, review) pairs so the note comes after a negative review about the same aspect

    Args:
        release_note_data (dict[int, dict]): Release notes
        review_data (dict[int, dict]): Reviews

    Returns:
        list[tuple[dict, dict]]: List of (release note, review) pairs that match the filtering criteria
    """
     
    # Filter to only negative reviews and reviews with at least 5 words
    review_data = {idx: data for idx, data in review_data.items() if data['sentiment'] == 0 and len(data['tokens']) >= 5}

    filtered_pairs = []
    for note, review in product(release_note_data.values(), review_data.values()):
        logger.debug(
            f"Release note (aspect: {note['aspect']}, date: {note['date']}), "
            f"Review (aspect: {review['aspect']}, date: {review['at']})"
        )

        cosine = torch.cosine_similarity(
            torch.tensor(note['embedding']),
            torch.tensor(review['embedding']),
            dim=0
        ).item()

        # if note['aspect'] != review['aspect'] or not OVERLAP_KEYWORDS_FOR_RELEASE_FILTERING.intersection(
        #     set(note['tokens']),
        #     set(review['tokens']),
        # ):
        if note['aspect'] == review['aspect']:
            if cosine < 0.15:
                continue
        elif cosine < 0.4:
            continue

        note_date = datetime.strptime(note['date'], "%d %B %Y").date()
        review_date = datetime.strptime(review['at'], "%Y-%m-%d %H:%M:%S").date()

        if note_date > review_date:
            filtered_pairs.append((note, review))

    return filtered_pairs

def lcs_length(a: list[str], b: list[str]) -> int:
    """Calculate the length of the longest common subsequence between two lists of tokens

    Args:
        a (list[str]): First list of tokens
        b (list[str]): Second list of tokens

    Returns:
        int: Length of the longest common subsequence
    """
    matcher = SequenceMatcher(None, a, b)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return match.size

def score_pairs(filtered_pairs: list[tuple[dict, dict]]) -> dict[tuple[int, int], dict]:
    """Use cosine similarity to score and rank the filtered (release note, review) pairs based on their embeddings

    Args:
        filtered_pairs (list[tuple[dict, dict]]): (release note, review) pairs that have been filtered to match the criteria of the note coming after a negative review about the same aspect

    Returns:
        dict[tuple[int, int], dict]: Dict from (release_note_id, review_id) to {"similarity": float, "release_note": dict, "review": dict}) tuples
    """

    results = dict()
    for note, review in filtered_pairs:
        cosine = torch.cosine_similarity(
            torch.tensor(note['embedding']),
            torch.tensor(review['embedding']),
            dim=0
        ).item()

        lcs_length_value = lcs_length(note["tokens"], review["tokens"])

        lcs_score = lcs_length_value / max(1, min(len(note['tokens']), len(review['tokens'])))

        # Give a boost to pairs with long lcs
        similarity = 0.5 * cosine + 0.5 * lcs_score + (0.2 if lcs_length_value >= 3 else 0) + (0.3 if OVERLAP_KEYWORDS_FOR_RELEASE_FILTERING.intersection(set(note["tokens"]), set(review["tokens"])) else 0)

        # Maybe add a penalty for generic reviews

        results[(note['version'], review['reviewId'])] = {
            "similarity": similarity,
            "release_note": note,
            "review": review,
            "lcs_length": lcs_length_value,
        }

    return results

def write_results_to_json(sorted_results: list[tuple[tuple[int, int], dict]], total_candidate_pairs: int, output_file: str) -> None:
    """Formats and writes the results of the similarity comparison by putting the 10 best and worst matches into the json 

    Args:
        sorted_results (list[tuple[tuple[int, int], dict]]): Sorted list of ((release_note_id, review_id), {"similarity": float, "release_note": dict, "review": dict}) tuples, ranked by similarity in descending order
        total_candidate_pairs (int): Total number of release note-review pairs before filtering by aspect and date, used for context in the output stats
        total_filtered_pairs (int): Total number of release note-review pairs after filtering by aspect and date, used for context in the output stats
        output_file (str): Output file path for the results JSON
    """

    total_results = len(sorted_results)

    aspect_labels = load_aspect_labels()
    # Format results for JSON output
    top_k_results = []
    for idx, ((note_id, review_id), result) in enumerate(sorted_results[:10], 1):
        note = result["release_note"]
        review = result["review"]
        top_k_results.append({
            "rank": idx,
            "release_version": note_id,
            "review_id": review_id,
            "release_note": f"({aspect_labels[note['aspect']]}) {note['content']}",
            "review": f"({aspect_labels[review['aspect']]}) {review['content']}",
            "similarity": round(result["similarity"], 4),
            "lcs_length": result["lcs_length"],
        })

    bottom_k_results = []
    for idx, ((note_id, review_id), result) in enumerate(sorted_results[-10:], 1):
        note = result["release_note"]
        review = result["review"]
        bottom_k_results.append({
            "rank": total_results - 10 + idx,
            "release_version": note_id,
            "review_id": review_id,
            "release_note": f"({aspect_labels[note['aspect']]}) {note['content']}",
            "review": f"({aspect_labels[review['aspect']]}) {review['content']}",
            "similarity": round(result["similarity"], 4),
            "lcs_length": result["lcs_length"],
        })

    output_payload = {
        "top_matches": top_k_results,
        "bottom_matches": bottom_k_results,
        "stats": {
            "total_pairs": total_results,
            "pairs_before_aspect": total_candidate_pairs,
        }
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_payload, f, indent=2)

def release_notes_vs_reviews_comparison(model: AspectSentimentExtractor, output_file: str = "results/release_notes_comparison.json"):
    """Conduct comparison between release notes and reviews for the same app

    Args:
        model (AspectSentimentExtractor): Model for generating embeddings and aspect/sentiment predictions 
        output_file (str, optional): Output file to store results. Defaults to "results/release_notes_comparison.json".
    """

    logger.info("Loading release notes from CSV...")
    release_note_data = encode_text(model, load_csv_rows('datasets/slack_release_notes.csv'))

    logger.info("Loading reviews from CSV...")
    review_data = encode_text(model, load_csv_rows('datasets/slack_reviews.csv'), is_review=True)

    filtered_pairs = filter_pairs(release_note_data, review_data)
    total_candidate_pairs = len(release_note_data) * len(review_data)

    logger.info(f"Total release note-review pairs: {total_candidate_pairs}")
    logger.info(f"Filtered release note-review pairs (matching aspects): {len(filtered_pairs)}")

    scored_results = score_pairs(filtered_pairs)

    # Sort results by similarity in descending order
    sorted_results = sorted(scored_results.items(), key=lambda x: x[1]['similarity'], reverse=True)
    
    write_results_to_json(sorted_results, total_candidate_pairs, output_file)
    
    logger.info(f"Results written to {output_file}")


if __name__ == "__main__":
    model = AspectSentimentExtractor(DISTILBERT_BASE, num_aspects=12).to(DEVICE)
    model.load_state_dict(torch.load("./models/aspect_sentiment_extractor.pth", map_location=DEVICE))
    model.eval()

    release_notes_vs_reviews_comparison(model, output_file="results/release_notes_comparison.json")