from review import Review, Sentence

def test_review_from_dict_parses_values_and_types():
    row = {
		"domain": "productivity",
		"app": "things-3",
		"review_id": "69d44a5e-218f-4f55-8a99-6cca55d43ca1",
		"title": "Incredible Planner for Students",
		"review": "I was originally skeptical on paying $7 on a todo list app, but I have come to realize how great of an investment this was.",
		"rating": "5",
		"is_opinion": "TRUE",
	}

    review = Review.from_dict(row)

    assert review.domain == "productivity"
    assert review.app == "things-3"
    assert review.review_id == "69d44a5e-218f-4f55-8a99-6cca55d43ca1"
    assert review.title == "Incredible Planner for Students"
    assert review.review == "I was originally skeptical on paying $7 on a todo list app, but I have come to realize how great of an investment this was."
    assert review.rating == 5
    assert review.is_opinion is True


def test_sentence_from_dict_parses_values_and_types():
    review = Review.from_dict(
        {
            "domain": "productivity",
            "app": "things-3",
            "review_id": "69d44a5e-218f-4f55-8a99-6cca55d43ca1",
            "title": "Incredible Planner for Students",
            "review": "I was originally skeptical on paying $7 on a todo list app, but I have come to realize how great of an investment this was.",
            "rating": "5",
            "is_opinion": "TRUE",
        }
    )

    row = {
        "sentence_id": "014a7d01-f6c0-408a-897b-f6b36cdd8543",
        "review_id": "69d44a5e-218f-4f55-8a99-6cca55d43ca1",
        "category": "effectiveness",
        "term": "functionality",
        "from": "40",
        "to": "53",
        "sentiment": "positive",
    }

    sentence = Sentence.from_review_and_dict(review, row)

    assert sentence.sentence_id == "014a7d01-f6c0-408a-897b-f6b36cdd8543"
    assert sentence.review.review_id == "69d44a5e-218f-4f55-8a99-6cca55d43ca1"
    assert sentence.category == "effectiveness"
    assert sentence.term == "functionality"
    assert sentence.from_word == 40
    assert sentence.to_word == 53
    assert sentence.sentiment == "positive"