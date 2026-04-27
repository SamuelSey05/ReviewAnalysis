from src.data_models import TrainingReview, TrainingOpinion
from tests.constants import OPINION_REVIEW_ROW, SENTENCE_ROW

def test_review_from_dict_parses_values_and_types():
    review = TrainingReview.from_dict(OPINION_REVIEW_ROW)

    assert review.domain == "productivity"
    assert review.app == "things-3"
    assert review.review_id == "69d44a5e-218f-4f55-8a99-6cca55d43ca1"
    assert review.title == "Incredible Planner for Students"
    assert review.review == "I was originally skeptical on paying $7 on a todo list app, but I have come to realize how great of an investment this was."
    assert review.rating == 5
    assert review.is_opinion is True


def test_opinion_from_dict_parses_values_and_types():
    review = TrainingReview.from_dict(OPINION_REVIEW_ROW)
    opinion = TrainingOpinion.from_review_and_dict(review, SENTENCE_ROW)

    assert opinion.sentence_id == "014a7d01-f6c0-408a-897b-f6b36cdd8543"
    assert opinion.review.review_id == "69d44a5e-218f-4f55-8a99-6cca55d43ca1"
    assert opinion.category == "effectiveness"
    assert opinion.term == "functionality"
    assert opinion.from_word == 40
    assert opinion.to_word == 53
    assert opinion.sentiment == "positive"