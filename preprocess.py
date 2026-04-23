import csv
import json
import logging
from pathlib import Path

from training_review import TrainingReview, TrainingOpinion

logger = logging.getLogger(__name__)

def load_csv(file_path: str) -> tuple[dict[str, TrainingReview], list[TrainingOpinion]]:
    """Load a csv file and return its contents as a list of dictionaries.
    
    Args:
        file_path (str): Path to the csv file.
        
    Returns:
        tuple[dict[str, TrainingReview], list[TrainingOpinion]]: A tuple containing:
            - A dictionary mapping review IDs to TrainingReview objects.
            - A list of TrainingOpinion objects for rows marked as opinions.
    """
    
    logger.info(f"Loading dataset from {file_path}...")

    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        reviews = dict()
        opinions = []
        for row in csv_reader:
            review = TrainingReview.from_dict(row)
            if review.review_id not in reviews:
                reviews[review.review_id] = review
            if review.is_opinion:
                # Opinions are only stored for reviews that contain opinions
                opinions.append(TrainingOpinion.from_review_and_dict(review, row))

    return reviews, opinions

def load_aspect_labels(path: str = "./resources/aspect_labels.json") -> list[str]:
    """Load aspect category labels from a JSON file.

    Args:
        path (str): Path to the JSON file containing aspect labels. Defaults to "./resources/aspect_labels.json".

    Returns:
        list[str]: List of aspect category labels.
    """
    
    labels_path = Path(path)
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)