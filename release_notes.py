from collections import Counter
import csv
from datetime import datetime, date, timedelta
from itertools import product
from difflib import SequenceMatcher
import json
import logging
import os
import re
import argparse

from sentence_transformers import SentenceTransformer, util

import numpy as np
import torch

from aspect_based import AspectSentimentExtractor, pool_embeddings
from comparison_models import EncodedReleaseNote, EncodedReview, NoteReviewPair, PairKey, PairResult, RankedResult
from config import DISTILBERT_BASE, DEVICE, BATCH_SIZE, OVERLAP_KEYWORDS_FOR_RELEASE_FILTERING
from preprocess import load_aspect_labels
from processing import tokenize
import release_notes_plots

logger = logging.getLogger(__name__)


def load_csv_rows(file_path: str) -> list[dict[str, str]]:
    """Loads CSV rows into a list of dictionaries.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list[dict[str, str]]: CSV rows keyed by column name.
    """

    with open(file_path, 'r') as file:
        return list(csv.DictReader(file))
    
def extract_keyword_tokens(text: str) -> list[str]:
    """Extract normalized word-like tokens from text.

    Args:
        text (str): Input text.

    Returns:
        list[str]: Lower-cased alphanumeric tokens including apostrophes.
    """

    return re.findall(r"[a-z0-9']+", text.lower())


def _collect_encoded_rows(
    model: AspectSentimentExtractor | SentenceTransformer,
    rows: list[dict[str, str]],
) -> list[tuple[int, dict[str, str], np.ndarray, int, int]]:
    """Encode rows of either reviews or release notes using model provided.

    Args:
        model (AspectSentimentExtractor): Model to use for encoding rows, used to generate embeddings and aspect/sentiment predictions.
        rows (list[dict[str, str]]): Rows from csv files to encode, as list of dictionaries. Each dictionary should contain a "content" key with the text to encode.

    Returns:
        list[tuple[int, dict[str, str], np.ndarray, int, int]]: List of tuples containing (row index, row data, embedding, aspect prediction, sentiment prediction) for each row in the input list.
    """

    encoded_rows: list[tuple[int, dict[str, str], np.ndarray, int, int]] = []

    with torch.no_grad():
        for i in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[i:i+BATCH_SIZE]

            if isinstance(model, AspectSentimentExtractor):
                inputs = tokenize([row["content"] for row in batch_rows], DISTILBERT_BASE)
                input_ids = inputs['input_ids'].to(DEVICE)
                attention_mask = inputs['attention_mask'].to(DEVICE)

                outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                mean_pooled = pool_embeddings(outputs.last_hidden_state, attention_mask)
                embeddings = mean_pooled.detach().cpu().numpy()

                aspects, sentiments = model.aspect_sentiment_inference(
                    input_ids=input_ids,
                    attention_masks=attention_mask,
                    batch_size=BATCH_SIZE,
                )
            else:
                embeddings = model.encode([row["content"] for row in batch_rows], convert_to_numpy=True, normalize_embeddings=True)
                aspects = [0] * len(batch_rows) 
                sentiments = [0] * len(batch_rows)

            for batch_idx, row in enumerate(batch_rows):
                encoded_rows.append((
                    i + batch_idx,
                    row,
                    embeddings[batch_idx],
                    aspects[batch_idx],
                    sentiments[batch_idx],
                ))

    return encoded_rows

def encode_release_notes(model: AspectSentimentExtractor | SentenceTransformer, rows: list[dict[str, str]]) -> dict[int, EncodedReleaseNote]:
    """Encode release-note rows into typed release-note objects.

    Args:
        model (AspectSentimentExtractor | SentenceTransformer): Encoder model used to build embeddings and optional predictions.
        rows (list[dict[str, str]]): Release-note CSV rows.

    Returns:
        dict[int, EncodedReleaseNote]: Mapping from row index to encoded release-note object.
    """
    data: dict[int, EncodedReleaseNote] = {}

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


def encode_reviews(model: AspectSentimentExtractor | SentenceTransformer, rows: list[dict[str, str]]) -> dict[int, EncodedReview]:
    """Encode reviews into the EncodedReview object with embeddings and aspect/sentiment predictions from the model.

    Args:
        model (AspectSentimentExtractor): Model to use to generate embeddings and aspect/sentiment predictions for the reviews, should be the same model used to encode release notes for consistency
        rows (list[dict[str, str]]): Rows from the reviews CSV file, as list of dictionaries. Each dictionary should contain a "content" key with the review text, and optionally a "score" key with the review rating.

    Returns:
        dict[int, EncodedReview]: Dictionary mapping from row index to EncodedReview object containing the review data, embedding, aspect prediction, and sentiment prediction.
    """

    data: dict[int, EncodedReview] = {}

    for idx, row, embedding, aspect, sentiment in _collect_encoded_rows(model, rows):
        raw_score = row.get("score", "")
        data[idx] = EncodedReview(
            review_id=row.get("reviewId", ""),
            content=row.get("content", ""),
            at=row.get("at", ""),
            score=int(raw_score) if raw_score and raw_score.isdigit() else 0,
            reply_content=row.get("replyContent", ""),
            replied_at=row.get("repliedAt", ""),
            embedding=embedding,
            aspect=aspect,
            tokens=extract_keyword_tokens(row["content"]),
            sentiment=sentiment,
        )

    return data
    
def pair_cosine_similarity(note: EncodedReleaseNote, review: EncodedReview) -> float:
    """Compute cosine similarity between one release note and one review embedding.

    Args:
        note (EncodedReleaseNote): Encoded release note.
        review (EncodedReview): Encoded review.

    Returns:
        float: Cosine similarity score.
    """

    return torch.nn.functional.cosine_similarity(
        torch.tensor(note.embedding),
        torch.tensor(review.embedding),
        dim=0,
    ).item()

def pair_dates(note: EncodedReleaseNote, review: EncodedReview) -> tuple[date, date]:
    """Parse and return comparable release-note and review dates.

    Args:
        note (EncodedReleaseNote): Encoded release note with DD Month YYYY date.
        review (EncodedReview): Encoded review with timestamp date.

    Returns:
        tuple[date, date]: Parsed (note_date, review_date).
    """

    note_date = datetime.strptime(note.date, "%d %B %Y").date()
    review_date = datetime.strptime(review.at, "%Y-%m-%d %H:%M:%S").date()
    return note_date, review_date

def filter_pairs(
    release_note_data: dict[int, EncodedReleaseNote],
    negative_reviews: dict[int, EncodedReview],
) -> list[NoteReviewPair]:
    """Filters (release note, review) pairs so the note comes after a negative review about the same aspect

    Args:
        release_note_data (dict[int, EncodedReleaseNote]): Release notes
        negative_reviews (dict[int, EncodedReview]): Negative reviews, filtered to only include those with at least 5 tokens to avoid generic one-word reviews

    Returns:
        list[NoteReviewPair]: List of (release note, review) pairs that match the filtering criteria
    """

    filtered_pairs = []
    for note, review in product(release_note_data.values(), negative_reviews.values()):
        logger.debug(
            f"Release note (aspect: {note.aspect}, date: {note.date}), "
            f"Review (aspect: {review.aspect}, date: {review.at})"
        )

        cosine = pair_cosine_similarity(note, review)

        # if note['aspect'] != review['aspect'] or not OVERLAP_KEYWORDS_FOR_RELEASE_FILTERING.intersection(
        #     set(note['tokens']),
        #     set(review['tokens']),
        # ):
        if note.aspect == review.aspect:
            if cosine < 0.15:
                continue
        elif cosine < 0.4:
            continue

        note_date, review_date = pair_dates(note, review)

        # Release note must come after the review, and within a year of the review
        if note_date > review_date and (note_date - review_date) < timedelta(days=365):
            filtered_pairs.append((note, review))

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

def score_pairs(filtered_pairs: list[NoteReviewPair]) -> dict[PairKey, PairResult]:
    """Use cosine similarity to score and rank the filtered (release note, review) pairs based on their embeddings

    Args:
        filtered_pairs (list[NoteReviewPair]): (release note, review) pairs that have been filtered to match the criteria of the note coming after a negative review about the same aspect

    Returns:
        dict[PairKey, PairResult]: Dict from (release_note_id, review_id) to typed pair metrics.
    """

    results: dict[PairKey, PairResult] = {}
    for note, review in filtered_pairs:
        cosine = pair_cosine_similarity(note, review)

        longest_match = get_longest_match(note.tokens, review.tokens)
        longest_match_length = len(longest_match)

        longest_match_score = longest_match_length / max(1, min(len(note.tokens), len(review.tokens)))

        # Give a boost to pairs with long lcs
        similarity = 0.4 * cosine + 0.4 * longest_match_score + (0.2 if set(longest_match).intersection(OVERLAP_KEYWORDS_FOR_RELEASE_FILTERING) else 0)

        # Maybe add a penalty for generic reviews

        note_date, review_date = pair_dates(note, review)

        results[(note.release_note_id, review.review_id)] = PairResult(
            similarity=similarity,
            release_note=note,
            review=review,
            lcs_length=longest_match_length,
            time_diff_days=(note_date - review_date).days,
        )

    return results

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
            "lcs_length": result.lcs_length,
            "time_diff_days": result.time_diff_days,
        })

    return rows

def write_results_to_json(sorted_results: list[RankedResult], total_candidate_pairs: int, no_of_negative_reviews: int, output_file: str, k: int = 10) -> None:
    """Formats and writes the results of the similarity comparison to JSON, including top/bottom k matches and aggregated stats.

    Args:
        sorted_results (list[RankedResult]): Sorted list of ((release_note_id, review_id), PairResult) tuples, ranked by similarity in descending order
        total_candidate_pairs (int): Total number of release note-review pairs before filtering by aspect and date, used for context in the output stats
        no_of_negative_reviews (int): Total number of negative reviews, used for calculating match density in the output stats
        output_file (str): Output file path for the results JSON
        k (int, optional): Number of top and bottom results to include in the output. Defaults to 10.
    """

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

    density, mttr = calculate_reactivity(sorted_results, no_of_negative_reviews=no_of_negative_reviews)

    output_payload = {
        "top_matches": top_k_results,
        "bottom_matches": bottom_k_results,
        "stats": {
            "total_pairs": total_results,
            "pairs_before_filter": total_candidate_pairs,
            "match_density": density,
            "mean_time_to_resolution_days": mttr,
            "times_to_resolutions": [pair_result.time_diff_days for _, pair_result in sorted_results if pair_result.similarity > 0.5],
        }
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_payload, f, indent=2)

def write_aspect_based_metrics(sorted_results: list[RankedResult], aspect_counts: dict, output_file: str) -> None:
    """Calculate and write per-aspect match density and mean time-to-resolution metrics to JSON and PNG.

    Args:
        sorted_results (list[RankedResult]): Sorted list of ((release_note_id, review_id), PairResult) tuples
        aspect_counts (dict): Dictionary mapping aspect IDs to their total count in negative reviews
        output_file (str): Output file path for the metrics JSON (PNG will be generated with .png suffix)
    """
    aspect_metrics = dict()

    for (release_note_id, review_id), result in sorted_results:
        review: EncodedReview = result.review
        aspect = review.aspect
        if aspect not in aspect_metrics:
            aspect_metrics[aspect] = {
                "fulfilments": 0,
                "total": aspect_counts.get(aspect, 0),
                "time_diffs": [],
            }

        if result.similarity > 0.5:
            aspect_metrics[aspect]["fulfilments"] += 1
            aspect_metrics[aspect]["time_diffs"].append(result.time_diff_days)
            

    for aspect, metrics in aspect_metrics.items():
        metrics["match_density"] = metrics["fulfilments"] / metrics["total"] if metrics["total"] > 0 else 0
        metrics["mean_time_to_resolution_days"] = sum(metrics["time_diffs"]) / len(metrics["time_diffs"]) if metrics["time_diffs"] else None

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(aspect_metrics, f, indent=2)

    release_notes_plots.plot_aspect_density_comparison(
        aspect_metrics,
        aspect_labels=load_aspect_labels(),
        output_file=output_file.replace(".json", ".png"),
    )


def write_results(sorted_results: list[RankedResult], total_candidate_pairs: int, no_of_negative_reviews: int, aspect_counts: dict, output_file: str, k: int = 10) -> None:
    """Write both raw and deduplicated result variants to JSON files.

    Args:
        sorted_results (list[RankedResult]): Sorted list of results by similarity (descending)
        total_candidate_pairs (int): Total candidate pairs before filtering
        no_of_negative_reviews (int): Number of negative reviews
        aspect_counts (dict): Counts of aspects in the negative reviews, used for calculating aspect-based MTTR and density
        output_file (str): Output file path for raw results
        k (int, optional): Number of top and bottom results to include. Defaults to 10.
    """
    # Write raw results
    write_results_to_json(sorted_results, total_candidate_pairs, no_of_negative_reviews, output_file, k=k)
    
    # Write deduplicated results
    deduped_sorted_results = dedup_results_by_release_note(sorted_results)
    deduped_output_file = output_file.replace(".json", "_dedup.json")
    write_results_to_json(deduped_sorted_results, total_candidate_pairs, no_of_negative_reviews, deduped_output_file, k=k)

    # Write aspect-based MTTR and density
    write_aspect_based_metrics(sorted_results, aspect_counts, output_file.replace(".json", "_aspect_metrics.json"))

def calculate_reactivity(sorted_results: list[RankedResult], no_of_negative_reviews: int, threshold: float = 0.5) -> tuple[float, float | None]:
    """Calculate match density and mean time-to-resolution above a similarity threshold.

    Args:
        sorted_results (list[RankedResult]): Ranked pair results.
        no_of_negative_reviews (int): Number of negative reviews considered.
        threshold (float, optional): Similarity threshold for counting fulfilments. Defaults to 0.5.

    Returns:
        tuple[float, float | None]: (match_density, mean_time_to_resolution_days).
    """

    fulfilments = [pair_result for _, pair_result in sorted_results if pair_result.similarity > threshold]

    if not fulfilments:
        return 0, None

    density = len(fulfilments) / no_of_negative_reviews

    mttr = sum(result.time_diff_days for result in fulfilments) / len(fulfilments)

    return density, mttr

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


def release_notes_vs_reviews_comparison(
    model: AspectSentimentExtractor | SentenceTransformer,
    app_name: str,
    output_file: str = "results/release_notes_comparison.json",
) -> tuple[list[RankedResult], int, int]:
    """Conduct comparison between release notes and reviews for the same app

    Args:
        model (AspectSentimentExtractor): Model for generating embeddings and aspect/sentiment predictions 
        app_name (str): Name of the app to compare release notes and reviews for, used to load the correct CSV files
        output_file (str, optional): Output file to store results. Defaults to "results/release_notes_comparison.json".    
    Returns:
        list[RankedResult]: Sorted list of ((release_note_id, review_id), PairResult) tuples, ranked by similarity in descending order
        int: Total number of release note-review pairs before filtering by aspect and date, used for context in the output stats
        int: Total number of negative reviews, used for calculating match density in the output stats

    """

    logger.info("Loading release notes from CSV...")
    release_note_data = encode_release_notes(model, load_csv_rows(f'datasets/{app_name}_release_notes.csv'))

    logger.info("Loading reviews from CSV...")
    review_data = encode_reviews(model, load_csv_rows(f'datasets/{app_name}_reviews.csv'))

    # Filter to only negative reviews and reviews with at least 5 words
    negative_reviews = {idx: data for idx, data in review_data.items() if data.sentiment == 0 and len(data.tokens) >= 5}

    aspect_counts = Counter(data.aspect for data in negative_reviews.values())

    non_generic_notes = {idx: data for idx, data in release_note_data.items() if len(data.tokens) >= 5}

    filtered_pairs = filter_pairs(non_generic_notes, negative_reviews)
    total_candidate_pairs = len(release_note_data) * len(review_data)

    logger.info(f"Total release note-review pairs: {total_candidate_pairs}")
    logger.info(f"Filtered release note-review pairs (matching aspects): {len(filtered_pairs)}")

    scored_results = score_pairs(filtered_pairs)

    # Sort results by similarity in descending order
    sorted_results = sorted(scored_results.items(), key=lambda x: x[1].similarity, reverse=True)

    write_results(sorted_results, total_candidate_pairs, len(negative_reviews), aspect_counts, output_file, k=30)
    
    logger.info(f"Results written to {output_file} and {output_file.replace('.json', '_deduped.json')}")

    return sorted_results, total_candidate_pairs, len(negative_reviews)


def sbert_comparison() -> None:
    """Run SBERT-based similarity comparison and write top/bottom 30 pairs to file.
    
    Encodes all release notes and reviews using SentenceTransformer all-MiniLM-L6-v2,
    computes cosine similarities between all pairs, and outputs top and bottom 30 matches
    with full text content to results/sbert_comparison_top_bottom.txt.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    release_note_data = load_csv_rows('datasets/discord_release_notes.csv')
    review_data = load_csv_rows('datasets/discord_reviews.csv')

    logger.info(
        "Encoding %d release notes and %d reviews for SBERT comparison...",
        len(release_note_data),
        len(review_data),
    )

    review_embeddings = model.encode([row["content"] for row in review_data], convert_to_tensor=True, normalize_embeddings=True)
    release_note_embeddings = model.encode([row["content"] for row in release_note_data], convert_to_tensor=True, normalize_embeddings=True)

    # Keep matrix operations vectorized; avoid per-pair .item() and Python object creation.
    cosine_scores = util.cos_sim(release_note_embeddings, review_embeddings).cpu().numpy()

    n_notes, n_reviews = cosine_scores.shape
    total_pairs = n_notes * n_reviews
    k = min(30, total_pairs)
    flat_scores = cosine_scores.ravel()

    top_idx = np.argpartition(flat_scores, -k)[-k:]
    top_idx = top_idx[np.argsort(flat_scores[top_idx])[::-1]]

    bottom_idx = np.argpartition(flat_scores, k - 1)[:k]
    bottom_idx = bottom_idx[np.argsort(flat_scores[bottom_idx])]

    os.makedirs("results", exist_ok=True)
    output_file = "results/sbert_comparison_top_bottom.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'Rank':<6} {'Release Note ID':<20} {'Review ID':<20} {'Similarity':<12}\\n")
        f.write("=" * 60 + "\\n")

        f.write(f"\\nTOP {k} PAIRS:\\n")
        f.write("-" * 60 + "\\n")
        for rank, flat_idx in enumerate(top_idx, start=1):
            note_idx, review_idx = divmod(int(flat_idx), n_reviews)
            note_row = release_note_data[note_idx]
            review_row = review_data[review_idx]
            note_id = note_row.get("release_note_id", f"n{note_idx}")
            review_id = review_row.get("reviewId", f"r{review_idx}")
            similarity = float(flat_scores[flat_idx])

            f.write(f"{rank:<6} {note_id:<20} {review_id:<20} {similarity:<12.4f}\\n")
            f.write(f"  Release Note: {note_row.get('content', '')}\\n")
            f.write(f"  Review:       {review_row.get('content', '')}\\n\\n")

        f.write(f"\\nBOTTOM {k} PAIRS:\\n")
        f.write("-" * 60 + "\\n")
        for offset, flat_idx in enumerate(bottom_idx):
            note_idx, review_idx = divmod(int(flat_idx), n_reviews)
            note_row = release_note_data[note_idx]
            review_row = review_data[review_idx]
            note_id = note_row.get("release_note_id", f"n{note_idx}")
            review_id = review_row.get("reviewId", f"r{review_idx}")
            similarity = float(flat_scores[flat_idx])
            rank = total_pairs - k + offset + 1

            f.write(f"{rank:<6} {note_id:<20} {review_id:<20} {similarity:<12.4f}\\n")
            f.write(f"  Release Note: {note_row.get('content', '')}\\n")
            f.write(f"  Review:       {review_row.get('content', '')}\\n\\n")

    logger.info("SBERT comparison complete. Wrote top/bottom %d pairs to %s", k, output_file)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Compare release notes and reviews for an app using SBERT or aspect-based model.")

    argparser.add_argument(
        "--use_sbert",
        action="store_true",
        help="Whether to use SBERT for the comparison instead of the aspect-based model. If not set, the aspect-based model will be used by default.",
    )
    argparser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Base directory for output files. Defaults to results/ for normal mode and results/sbert/ for SBERT mode.",
    )

    args = argparser.parse_args()

    default_results_dir = "results/sbert" if args.use_sbert else "results"
    results_dir = args.results_dir or default_results_dir
    os.makedirs(results_dir, exist_ok=True)

    if args.use_sbert:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        model = AspectSentimentExtractor(DISTILBERT_BASE, num_aspects=12).to(DEVICE)
        model.load_state_dict(torch.load("./models/aspect_sentiment_extractor.pth", map_location=DEVICE))
        model.eval()


    with open("./web_scraping/apps.csv", 'r') as f:
        reader = csv.DictReader(f)
        apps = [row['name'].strip().lower() for row in reader if row['app_id'].strip()]

    app_summaries = {}
    all_results = []
    for app_name in apps:
        logger.info(f"Comparing release notes and reviews for {app_name}...")
        comparison_output_file = os.path.join(results_dir, f"{app_name}_release_notes_comparison_thinned.json")
        sorted_results, total_candidate_pairs, no_of_negative_reviews = release_notes_vs_reviews_comparison(
            model,
            app_name,
            output_file=comparison_output_file,
        )

        with open(comparison_output_file, 'r') as f:
            data = json.load(f)

        app_summaries[app_name] = {
            "total_pairs": total_candidate_pairs,
            "match_density": data["stats"]["match_density"],
            "mean_time_to_resolution_days": data["stats"]["mean_time_to_resolution_days"],
            "times_to_resolutions": data["stats"]["times_to_resolutions"],
        }

        all_results.extend(sorted_results)

    # Plot graphs
    release_notes_plots.plot_mttr_comparison(
        app_summaries,
        output_file=os.path.join(results_dir, "mttr_comparison.png"),
    )

    release_notes_plots.plot_resolution_time_distribution(
        app_summaries,
        output_file=os.path.join(results_dir, "resolution_time_distribution.png"),
    )

    release_notes_plots.plot_specificity_comparison(
        all_results,
        output_file=os.path.join(results_dir, "similarity_vs_resolution_time.png"),
    )

    # sbert_comparison()
