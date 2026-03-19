import csv

from preprocess import load_csv
from tests.constants import CSV_FIELDNAMES, NON_OPINION_REVIEW_ROW, OPINION_REVIEW_ROW


def test_load_csv_collects_reviews_and_opinion_sentences(tmp_path):
    csv_path = tmp_path / "test.csv"

    rows = [OPINION_REVIEW_ROW, NON_OPINION_REVIEW_ROW]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    reviews, sentences = load_csv(str(csv_path))

    assert len(reviews) == 2
    assert len(sentences) == 1
    assert sentences[0].review.review_id == "69d44a5e-218f-4f55-8a99-6cca55d43ca1"
    assert reviews["e633e20a-07c1-4a5e-80b1-b104b6cf6a61"].is_opinion is False
    assert isinstance(reviews["69d44a5e-218f-4f55-8a99-6cca55d43ca1"].rating, int)
    assert reviews["69d44a5e-218f-4f55-8a99-6cca55d43ca1"].rating == 5
    assert isinstance(sentences[0].from_word, int)
    assert isinstance(sentences[0].to_word, int)
    assert sentences[0].from_word == 40
    assert sentences[0].to_word == 53