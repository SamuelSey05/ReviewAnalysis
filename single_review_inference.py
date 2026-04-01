import argparse
import logging

import torch

from aspect_based import AspectSentimentExtractor
from config import DEVICE, DISTILBERT_BASE
from processing import tokenize
from review import Review

logger = logging.getLogger(__name__)

SENTIMENT_LABELS = ["negatdive", "neutral", "positive"]
ASPECT_LABELS = ['learnability', 'aesthetics', 'general', 'cost', 'efficiency', 'safety', 'enjoyability', 'security', 'compatibility', 'effectiveness', 'usability', 'reliability']

def read_review_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def build_review(text: str) -> Review:
    return Review(
        domain="",
        app="",
        review_id="",
        title="",
        review=text,
        rating=0,
        is_opinion=True,
    )

def main() -> None:
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

    print(f"Aspect: {aspect_label} (index={aspect_idx})")
    print(f"Sentiment: {sentiment_label} (index={sentiment_idx})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
