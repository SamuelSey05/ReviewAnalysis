import csv

import torch

from aspect_based import AspectSentimentExtractor
from config import DISTILBERT_BASE, DEVICE
from main import map_rating_to_sentiment, prepare_aspect_dataset
from preprocess import load_csv
from processing import tokenize
from tests.helpers import use_deterministic_dataloader, weighted_dataset_loss
from trainer import train_aspect_sentiment_extractor
from tests.constants import CSV_FIELDNAMES, NON_OPINION_REVIEW_ROW, OPINION_REVIEW_ROW

def test_full_integration(monkeypatch, tmp_path):
    """Full integration test"""

    csv_path = tmp_path / "integration.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows([OPINION_REVIEW_ROW, NON_OPINION_REVIEW_ROW])

    reviews, sentences = load_csv(str(csv_path))
    assert len(reviews) == 2
    assert len(sentences) == 1

    review_list = list(reviews.values())
    review_inputs = tokenize([x.review for x in review_list], DISTILBERT_BASE)
    assert len(review_inputs["input_ids"]) == 2
    assert len(review_inputs["attention_mask"]) == 2

    true_sentiments = [map_rating_to_sentiment(float(review.rating)) for review in review_list]
    review_ids = [review.review_id for review in review_list]

    dataset, aspects, _, _ = prepare_aspect_dataset(
        sentences=sentences,
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

    before_aspect_weights = model.state_dict()["aspect_head.0.weight"].detach().clone()
    before_sentiment_weights = model.sentiment_head.weight.detach().clone()
    before_loss = weighted_dataset_loss(model, dataset, aspect_criterion, sentiment_criterion)

    train_aspect_sentiment_extractor(
        model=model,
        dataset=dataset,
        aspect_criterion=aspect_criterion,
        sentiment_criterion=sentiment_criterion,
        num_epochs=10,
        device=DEVICE,
    )

    after_aspect_weights = model.state_dict()["aspect_head.0.weight"].detach().clone()
    after_sentiment_weights = model.sentiment_head.weight.detach().clone()
    after_loss = weighted_dataset_loss(model, dataset, aspect_criterion, sentiment_criterion)

    assert not torch.equal(before_aspect_weights, after_aspect_weights)
    assert not torch.equal(before_sentiment_weights, after_sentiment_weights)

    assert after_loss.item() < before_loss.item()
