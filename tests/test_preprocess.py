import csv

from preprocess import load_csv


def test_load_csv_collects_reviews_and_opinion_sentences(tmp_path):
    csv_path = tmp_path / "test.csv"

    fieldnames = [
        "domain",
        "app",
        "review_id",
        "title",
        "review",
        "rating",
        "is_opinion",
        "sentence_id",
        "category",
        "term",
        "from",
        "to",
        "sentiment",
    ]

    rows = [
        {
            "domain": "productivity",
            "app": "things-3",
            "review_id": "69d44a5e-218f-4f55-8a99-6cca55d43ca1",
            "title": "Incredible Planner for Students",
            "review": "I was originally skeptical on paying $7 on a todo list app, but I have come to realize how great of an investment this was.",
            "rating": "5",
            "is_opinion": "TRUE",
            "sentence_id": "014a7d01-f6c0-408a-897b-f6b36cdd8543",
            "category": "effectiveness",
            "term": "functionality",
            "from": "40",
            "to": "53",
            "sentiment": "positive",
        },
        {
            "domain": "productivity",
            "app": "notability",
            "review_id": "e633e20a-07c1-4a5e-80b1-b104b6cf6a61",
            "title": "Great app",
            "review": "I have been using this app for over 3 years now and it has proven over and over again to deliver.",
            "rating": "5",
            "is_opinion": "FALSE",
            "sentence_id": "00a8d4a4-9c8e-4d1c-9085-ffd1f62ae039",
            "category": "N/A",
            "term": "N/A",
            "from": "N/A",
            "to": "N/A",
            "sentiment": "negative",
        },
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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