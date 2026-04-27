import csv
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
import itertools
import json
import logging
import os

import numpy as np
import torch

from src.config import OVERLAP_KEYWORDS_FOR_RELEASE_FILTERING
from src.data_models import EncodedReleaseNote, EncodedReview, PairKey, PairResult, RankedResult, ReviewNotePair
from src.model_architecture import AspectSentimentModel
from src.processing import extract_keyword_tokens, load_aspect_labels, load_csv_rows
from src.release_notes_comparison_plots import plot_aspect_density_comparison

logger = logging.getLogger(__name__)

def get_app_names() -> list[str]:
    """Get the list of app names from the csv in web_scraping/

    Returns:
        list[str]: List of app names to do release notes comparison for.
    """
    
    with open("./web_scraping/apps.csv", 'r') as f:
        reader = csv.DictReader(f)
        apps = [row['name'].strip().lower() for row in reader if row['app_id'].strip()]

    return apps

def _collect_encoded_rows(
    model: AspectSentimentModel,
    rows: list[dict[str, str]],
) -> list[tuple[int, dict[str, str], np.ndarray, int, int]]:
    """Encode rows of either reviews or release notes using model provided.

    Args:
        model (AspectSentimentModel): Model to use for encoding rows, used to generate embeddings and aspect/sentiment predictions.
        rows (list[dict[str, str]]): Rows from csv files to encode, as list of dictionaries. Each dictionary should contain a "content" key with the text to encode.

    Returns:
        list[tuple[int, dict[str, str], np.ndarray, int, int]]: List of tuples containing (row index, row data, embedding, aspect prediction, sentiment prediction) for each row in the input list.
    """

    encoded_rows: list[tuple[int, dict[str, str], np.ndarray, int, int]] = []

    with torch.no_grad():
        texts = [row["content"] for row in rows]

        embeddings = model.get_embeddings(texts).cpu().numpy()
        aspects, sentiments = model.aspect_sentiment_inference(texts)
        
        for i, row in enumerate(rows):
            encoded_rows.append((
                i,
                row,
                embeddings[i],
                aspects[i],
                sentiments[i],
            ))

    return encoded_rows

def encode_release_notes(model: AspectSentimentModel, app_name: str) -> dict[int, EncodedReleaseNote]:
    """Encode release-note rows into typed release-note objects.

    Args:
        model (AspectSentimentModel): Encoder model used to build embeddings and optional predictions.
        app_name (str): Name of the app to load release notes for, used to find the correct CSV file in web_scraping/.

    Returns:
        dict[int, EncodedReleaseNote]: Mapping from row index to encoded release-note object.
    """
    data: dict[int, EncodedReleaseNote] = {}

    rows = load_csv_rows(f"./web_scraping/{app_name}_release_notes.csv")

    for idx, row, embedding, aspect, _ in _collect_encoded_rows(model, rows):
        data[idx] = EncodedReleaseNote(
            release_note_id=row.get("release_note_id", ""),
            version=row.get("version", ""),
            date=row.get("date", ""),
            content=row.get("content", ""),
            embedding=embedding,
            aspect=aspect,
            tokens=extract_keyword_tokens(row["content"]),
        )

    return data


def encode_reviews(model: AspectSentimentModel, app_name: str) -> dict[int, EncodedReview]:
    """Encode reviews into the EncodedReview object with embeddings and aspect/sentiment predictions from the model.

    Args:
        model (AspectSentimentModel): Model to use to generate embeddings and aspect/sentiment predictions for the reviews, should be the same model used to encode release notes for consistency
        app_name (str): Name of the app to load reviews for, used to find the correct CSV file in web_scraping/.

    Returns:
        dict[int, EncodedReview]: Dictionary mapping from row index to EncodedReview object containing the review data, embedding, aspect prediction, and sentiment prediction.
    """

    data: dict[int, EncodedReview] = {}

    rows = load_csv_rows(f"./web_scraping/{app_name}_reviews.csv")

    for idx, row, embedding, aspect, sentiment in _collect_encoded_rows(model, rows):
        raw_score = row.get("score", "")
        data[idx] = EncodedReview(
            review_id=row.get("reviewId", ""),
            content=row.get("content", ""),
            at=row.get("at", ""),
            score=int(raw_score) if raw_score and raw_score.isdigit() else 0,
            embedding=embedding,
            aspect=aspect,
            tokens=extract_keyword_tokens(row["content"]),
            sentiment=sentiment,
        )

    return data

def pair_cosine_similarity(review: EncodedReview, note: EncodedReleaseNote) -> float:
    """Compute cosine similarity between one release note and one review embedding.

    Args:
        review (EncodedReview): Encoded review.
        note (EncodedReleaseNote): Encoded release note.

    Returns:
        float: Cosine similarity score.
    """

    return torch.nn.functional.cosine_similarity(
        torch.tensor(review.embedding),
        torch.tensor(note.embedding),
        dim=0,
    ).item()

def pair_dates(review: EncodedReview, note: EncodedReleaseNote) -> tuple[date, date]:
    """Parse and return comparable release-note and review dates.

    Args:
        note (EncodedReleaseNote): Encoded release note with DD Month YYYY date.
        review (EncodedReview): Encoded review with timestamp date.

    Returns:
        tuple[date, date]: Parsed (note_date, review_date).
    """

    review_date = datetime.strptime(review.at, "%Y-%m-%d %H:%M:%S").date()
    note_date = datetime.strptime(note.date, "%d %B %Y").date()
    return review_date, note_date

def filter_pairs(
    negative_reviews: dict[int, EncodedReview],
    release_note_data: dict[int, EncodedReleaseNote],
) -> list[ReviewNotePair]:
    """Filters (release note, review) pairs so the note comes after a negative review about the same aspect

    Args:
        negative_reviews (dict[int, EncodedReview]): Negative reviews, filtered to only include those with at least 5 tokens to avoid generic short reviews
        release_note_data (dict[int, EncodedReleaseNote]): Release notes with at least 5 tokens to avoid generic short notes

    Returns:
        list[NoteReviewPair]: List of (release note, review) pairs that match the filtering criteria
    """

    filtered_pairs: list[ReviewNotePair] = []

    for review, note in itertools.product(negative_reviews.values(), release_note_data.values()):
        # Release note must come after the review, and within a year of the review
        review_date, note_date = pair_dates(review, note)

        if note_date > review_date and (note_date - review_date) < timedelta(days=365):
            cosine_similarity = pair_cosine_similarity(review, note)

            if review.aspect == note.aspect:
                if cosine_similarity < 0.15:
                    continue
            elif cosine_similarity < 0.4:
                continue

            filtered_pairs.append((review, note))       

    return filtered_pairs

def get_longest_match(a: list[str], b: list[str]) -> list[str]:
    """Calculate the length of the longest match between two lists of tokens

    Args:
        a (list[str]): First list of tokens
        b (list[str]): Second list of tokens

    Returns:
        list[str]: The longest match
    """

    matcher = SequenceMatcher(None, a, b)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return a[match.a: match.a + match.size]

def score_pairs(filtered_pairs: list[ReviewNotePair]) -> dict[PairKey, PairResult]:
    """Use cosine similarity to score and rank the filtered (release note, review) pairs based on their embeddings

    Args:
        filtered_pairs (list[NoteReviewPair]): (release note, review) pairs that have been filtered to match the criteria of the note coming after a negative review about the same aspect

    Returns:
        dict[PairKey, PairResult]: Dict from (release_note_id, review_id) to typed pair metrics.
    """

    scored_results: dict[PairKey, PairResult] = {}

    for review, note in filtered_pairs:
        cosine_similarity = pair_cosine_similarity(review, note)

        longest_match = get_longest_match(note.tokens, review.tokens)
        longest_match_length = len(longest_match)

        # Normlise to shortest of review and note to avoid bias to longer texts
        longest_match_score = longest_match_length / max(1, min(len(note.tokens), len(review.tokens)))

        # Similarity score calculation
        # TODO: Maybe loosen up here and allow intersection anywhere in text
        similarity_score = 0.4 * cosine_similarity + 0.4 * longest_match_score + (0.2 if set(longest_match).intersection(OVERLAP_KEYWORDS_FOR_RELEASE_FILTERING) else 0)

        review_date, note_date = pair_dates(review, note)

        scored_results[(review.review_id, note.release_note_id)] = PairResult(
            similarity=similarity_score,
            review=review,
            release_note=note,
            longest_match_length=longest_match_length,
            time_diff_days=(note_date - review_date).days,
        )

    return scored_results


def release_notes_vs_reviews_comparison(model: AspectSentimentModel, app_name: str) -> tuple[list[tuple[PairKey, PairResult]], int, int]:
    """Conduct comparison between release notes and reviews for the same app

    Args:
        model (AspectSentimentModel): Model for generating embeddings and aspect/sentiment predictions 
        app_name (str): Name of the app to compare release notes and reviews for, used to load the correct CSV files
        output_file (str, optional): Output file to store results. Defaults to "results/release_notes_comparison.json".    
    Returns:
        list[RankedResult]: Sorted list of ((release_note_id, review_id), PairResult) tuples, ranked by similarity in descending order
        int: Total number of release note-review pairs before filtering by aspect and date, used for context in the output stats
        int: Total number of negative reviews, used for calculating match density in the output stats

    """

    release_notes = encode_release_notes(model, app_name)
    reviews = encode_reviews(model, app_name)

    # Number of possible combinations before filtering
    total_candidate_pairs = len(release_notes) * len(reviews)

    negative_reviews = {idx: review for idx, review in reviews.items() if review.sentiment == 0 and len(review.tokens) >= 5}

    non_generic_notes = {idx: note for idx, note in release_notes.items() if len(note.tokens) >= 5}
    
    filtered_pairs = filter_pairs(negative_reviews, non_generic_notes)

    logger.info(f"Total release note-review pairs: {total_candidate_pairs}")
    logger.info(f"Filtered release note-review pairs (matching aspects): {len(filtered_pairs)}")

    scored_results = score_pairs(filtered_pairs)

    sorted_results = sorted(scored_results.items(), key=lambda x: x[1].similarity, reverse=True)

    return sorted_results, total_candidate_pairs, len(negative_reviews)

def format_result_rows(
    ranked_results: list[RankedResult],
    aspect_labels: list[str],
    start_rank: int,
) -> list[dict]:
    """Format ranked pair results into JSON-serializable rows.

    Args:
        ranked_results (list[RankedResult]): Ranked ((release_note_id, review_id), PairResult) tuples.
        aspect_labels (list[str]): Aspect label names indexed by class id.
        start_rank (int): Rank value for the first row in this slice.

    Returns:
        list[dict]: Formatted rows with identifiers, text, and metrics.
    """
    
    rows = []
    for offset, ((release_note_id, review_id), result) in enumerate(ranked_results):
        note: EncodedReleaseNote = result.release_note
        review: EncodedReview = result.review
        rows.append({
            "rank": start_rank + offset,
            "release_note_id": release_note_id,
            "release_version": note.version,
            "review_id": review_id,
            "release_note": f"({aspect_labels[note.aspect]}) {note.content}",
            "review": f"({aspect_labels[review.aspect]}) {review.content}",
            "similarity": round(result.similarity, 4),
            "lcs_length": result.longest_match_length,
            "time_diff_days": result.time_diff_days,
        })

    return rows

def calculate_reactivity(sorted_results: list[RankedResult], no_of_negative_reviews: int, threshold: float = 0.5) -> tuple[float, list[int]]:
    """Calculate match density and mean time-to-resolution above a similarity threshold.

    Args:
        sorted_results (list[RankedResult]): Ranked pair results.
        no_of_negative_reviews (int): Number of negative reviews considered.
        threshold (float, optional): Similarity threshold for counting fulfilments. Defaults to 0.5.

    Returns:
        tuple[float | list[int]]: (match_density, times_to_resolutions).
    """

    fulfilments = [pair_result for _, pair_result in sorted_results if pair_result.similarity > threshold]

    if not fulfilments:
        return 0, []

    density = len(fulfilments) / no_of_negative_reviews

    times_to_resolutions = [pair_result.time_diff_days for pair_result in fulfilments]


    return density, times_to_resolutions

def dedup_results_by_release_note(sorted_results: list[RankedResult]) -> list[RankedResult]:
    """Keep only the highest-ranked match per release note ID.

    Args:
        sorted_results (list[RankedResult]): Similarity-ranked list of ((release_note_id, review_id), result) tuples.

    Returns:
        list[RankedResult]: Deduplicated ranking with at most one entry per release_note_id.
    """

    seen_release_note_ids = set()
    deduped_results = []

    for (release_note_id, review_id), result in sorted_results:
        if release_note_id in seen_release_note_ids:
            continue

        seen_release_note_ids.add(release_note_id)
        deduped_results.append(((release_note_id, review_id), result))

    return deduped_results

def write_top_and_bottom_pairs_to_json(sorted_results: list[RankedResult], total_candidate_pairs: int, no_of_negative_reviews: int, output_file: str, deduplicate: bool = False, k: int = 10) -> tuple[float, list[int]]:
    """Formats and writes the results of the similarity comparison to JSON, including top/bottom k matches and aggregated stats.

    Args:
        sorted_results (list[RankedResult]): Sorted list of ((release_note_id, review_id), PairResult) tuples, ranked by similarity in descending order
        total_candidate_pairs (int): Total number of release note-review pairs before filtering by aspect and date, used for context in the output stats
        no_of_negative_reviews (int): Total number of negative reviews, used for calculating match density in the output stats
        output_file (str): Output file path for the results JSON
        deduplicate (bool, optional): Whether to deduplicate results to keep only the top match per release note. Defaults to False.
        k (int, optional): Number of top and bottom results to include in the output. Defaults to 10.

    Returns:
        tuple[float, list[int]]: (match_density, times_to_resolutions) calculated for the output stats
    """

    if deduplicate:
        sorted_results = dedup_results_by_release_note(sorted_results)

    total_results = len(sorted_results)

    aspect_labels = load_aspect_labels()
    # Format results for JSON output
    top_slice = sorted_results[:k]
    bottom_slice = sorted_results[-k:]

    top_k_results = format_result_rows(top_slice, aspect_labels, start_rank=1)
    bottom_k_results = format_result_rows(
        bottom_slice,
        aspect_labels,
        start_rank=max(1, total_results - 9),
    )

    density, times_to_resolutions = calculate_reactivity(sorted_results, no_of_negative_reviews=no_of_negative_reviews)


    output_payload = {
        "top_matches": top_k_results,
        "bottom_matches": bottom_k_results,
        "stats": {
            "total_pairs": total_results,
            "pairs_before_filter": total_candidate_pairs,
            "match_density": density,
            "mean_time_to_resolution_days": np.mean(times_to_resolutions),
            "times_to_resolutions": times_to_resolutions,
        }
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_payload, f, indent=2)

    return density, times_to_resolutions

def write_aspect_based_metrics(sorted_results: list[RankedResult], output_file: str) -> None:
    """Calculate and write per-aspect match density and mean time-to-resolution metrics to JSON and PNG.

    Args:
        sorted_results (list[RankedResult]): Sorted list of ((release_note_id, review_id), PairResult) tuples
        output_file (str): Output file path for the metrics JSON (PNG will be generated with .png suffix)
    """
    aspect_metrics = {}

    for (release_note_id, review_id), result in sorted_results:
        review: EncodedReview = result.review
        aspect = review.aspect
        if aspect not in aspect_metrics:
            aspect_metrics[aspect] = {
                "fulfilments": 0,
                "total": 0,
                "time_diffs": [],
            }

        aspect_metrics[aspect]["total"] += 1

        if result.similarity > 0.5:
            aspect_metrics[aspect]["fulfilments"] += 1
            aspect_metrics[aspect]["time_diffs"].append(result.time_diff_days)
            

    for aspect, metrics in aspect_metrics.items():
        metrics["match_density"] = metrics["fulfilments"] / metrics["total"] if metrics["total"] > 0 else 0
        metrics["mean_time_to_resolution_days"] = sum(metrics["time_diffs"]) / len(metrics["time_diffs"]) if metrics["time_diffs"] else None

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(aspect_metrics, f, indent=2)

    plot_aspect_density_comparison(
        aspect_metrics,
        aspect_labels=load_aspect_labels(),
        output_file=output_file.replace(".json", ".png"),
    )