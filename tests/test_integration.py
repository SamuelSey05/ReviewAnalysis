import csv

import torch

from src.model_architecture import AspectSentimentExtractor
from src.config import DISTILBERT_BASE, DEVICE
from src.processing import load_dataset_csv, map_rating_to_sentiment, prepare_aspect_dataset
from src.processing import tokenize
from tests.helpers import use_deterministic_dataloader
from src.trainer import train_aspect_sentiment_extractor, weighted_aspect_sentiment_loss
from tests.constants import CSV_FIELDNAMES, NON_OPINION_REVIEW_ROW, OPINION_REVIEW_ROW

def test_full_integration(monkeypatch, tmp_path):
    """Full integration test"""

    csv_path = tmp_path / "integration.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows([OPINION_REVIEW_ROW, NON_OPINION_REVIEW_ROW])

    reviews, opinions = load_dataset_csv(str(csv_path))
    assert len(reviews) == 2
    assert len(opinions) == 1

    review_list = list(reviews.values())
    review_inputs = tokenize([x.review for x in review_list], DISTILBERT_BASE)
    assert len(review_inputs["input_ids"]) == 2
    assert len(review_inputs["attention_mask"]) == 2

    true_sentiments = [map_rating_to_sentiment(float(review.rating)) for review in review_list]
    review_ids = [review.review_id for review in review_list]

    dataset, aspects, _, _ = prepare_aspect_dataset(
        opinions=opinions,
        review_ids=review_ids,
        review_inputs=review_inputs,
        true_sentiments=true_sentiments,
    )

    torch.manual_seed(0)
    use_deterministic_dataloader(monkeypatch)

    model = AspectSentimentExtractor(
        num_aspects=len(aspects),
        num_sentiments=3,
        model_name=DISTILBERT_BASE,
    ).to(DEVICE)

    aspect_criterion = torch.nn.CrossEntropyLoss()
    sentiment_criterion = torch.nn.CrossEntropyLoss()

    batch = dataset.with_format("torch")[:]

    before_aspect_weights = model.state_dict()["aspect_head.0.weight"].detach().clone()
    before_sentiment_weights = model.sentiment_head.weight.detach().clone()
    before_loss = weighted_aspect_sentiment_loss(model, batch, aspect_criterion, sentiment_criterion, DEVICE)

    train_aspect_sentiment_extractor(
        model=model,
        dataset=dataset,
        aspect_criterion=aspect_criterion,
        sentiment_criterion=sentiment_criterion,
        num_epochs=10,
    )

    after_aspect_weights = model.state_dict()["aspect_head.0.weight"].detach().clone()
    after_sentiment_weights = model.sentiment_head.weight.detach().clone()
    after_loss = weighted_aspect_sentiment_loss(model, batch, aspect_criterion, sentiment_criterion, DEVICE)

    assert not torch.equal(before_aspect_weights, after_aspect_weights)
    assert not torch.equal(before_sentiment_weights, after_sentiment_weights)

    assert after_loss.item() < before_loss.item()
