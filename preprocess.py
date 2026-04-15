import csv
import json
import logging
from pathlib import Path

from training_review import TrainingReview, Sentence

logger = logging.getLogger(__name__)

def load_csv(file_path: str) -> tuple[dict[str, TrainingReview], list[Sentence]]:
    """Load a csv file and return its contents as a list of dictionaries.
    
    Args:
        file_path (str): Path to the csv file.
        
    Returns:
        tuple[dict[str, TrainingReview], list[Sentence]]: A tuple containing:
            - A dictionary mapping review IDs to TrainingReview objects.
            - A list of Sentence objects for rows marked as opinions.
    """
    
    logger.info(f"Loading dataset from {file_path}...")

    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        data = dict()
        review_parts = []
        for row in csv_reader:
            review = TrainingReview.from_dict(row)
            if review.review_id not in data:
                data[review.review_id] = review
            if review.is_opinion:
                # Sentences are only stored for reviews that contain opinions
                review_parts.append(Sentence.from_review_and_dict(review, row))

    return data, review_parts

def load_aspect_labels(path: str = "./models/aspect_labels.json") -> list[str]:
    """Load aspect category labels from a JSON file.

    Args:
        path (str): Path to the JSON file containing aspect labels. Defaults to "./models/aspect_labels.json".

    Returns:
        list[str]: List of aspect category labels.
    """
    
    labels_path = Path(path)
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)