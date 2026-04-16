import argparse
import logging

import torch

from aspect_based import AspectSentimentExtractor
from config import DEVICE, DISTILBERT_BASE
from preprocess import load_aspect_labels
from processing import tokenize
from training_review import TrainingReview

logger = logging.getLogger(__name__)

SENTIMENT_LABELS = ["negative", "neutral", "positive"]

ASPECT_LABELS = load_aspect_labels()

def read_review_text(path: str) -> str:
    """Read review text from a file.

    Args:
        path (str): Path to the text file containing the review.

    Returns:
        str: The review text with leading/trailing whitespace removed.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def build_review(text: str) -> TrainingReview:
    """Build a TrainingReview object from raw review text.

    Args:
        text (str): The review text content.

    Returns:
        TrainingReview: A TrainingReview object with default/empty fields and the given text.
    """
    return TrainingReview(
        domain="",
        app="",
        review_id="",
        title="",
        review=text,
        rating=0,
        is_opinion=True,
    )

def main() -> None:
    """Main entry point for aspect and sentiment inference on a single review text file.

    Reads a review from a text file, builds a TrainingReview object, tokenizes it,
    and runs aspect/sentiment extraction using the loaded model.
    """
    parser = argparse.ArgumentParser(
        description="Tokenise and run aspect+sentiment inference on a single review."
    )
    parser.add_argument(
        "review_file",
        help="Path to a text file containing a single review.",
    )
    args = parser.parse_args()

    review_text = read_review_text(args.review_file)
    if not review_text:
        raise ValueError("Review text file is empty.")

    review = build_review(review_text)

    inputs = tokenize([review.review], DISTILBERT_BASE)

    model = AspectSentimentExtractor(DISTILBERT_BASE, num_aspects=len(ASPECT_LABELS)).to(DEVICE)
    model.load_state_dict(torch.load("./models/aspect_sentiment_extractor.pth", map_location=DEVICE))
    model.eval()

    aspect_predictions, sentiment_predictions = model.aspect_sentiment_inference(
        input_ids=inputs["input_ids"],
        attention_masks=inputs["attention_mask"],
        batch_size=1,
    )

    aspect_idx = int(aspect_predictions[0])
    sentiment_idx = int(sentiment_predictions[0])

    aspect_label = ASPECT_LABELS[aspect_idx] if 0 <= aspect_idx < len(ASPECT_LABELS) else "unknown"
    sentiment_label = (
        SENTIMENT_LABELS[sentiment_idx]
        if 0 <= sentiment_idx < len(SENTIMENT_LABELS)
        else "unknown"
    )

    logger.info("Aspect: %s (index=%d)", aspect_label, aspect_idx)
    logger.info("Sentiment: %s (index=%d)", sentiment_label, sentiment_idx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
