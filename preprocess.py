import csv
from typing import Tuple

from review import Review, Sentence

def load_csv(file_path) -> Tuple[list[Review], list[Sentence]]:
    """Load a csv file and return its contents as a list of dictionaries.
    
    Args:
        file_path (str): Path to the csv file.
        
    Returns:
        list[dict]: List of dictionaries for each row in the csv file."""
    

    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        data = dict()
        review_parts = []
        for row in csv_reader:
            review = Review.from_dict(row)
            if review.review_id not in data:
                data[review.review_id] = review
            if review.is_opinion:
                review_parts.append(Sentence.from_review_and_dict(review, row))

    return list(data.values()), review_parts