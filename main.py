import logging
from typing import Counter

import torch
from sklearn.metrics import accuracy_score, classification_report

from arguements import get_args
from aspect_based import AspectSentimentExtractor
from config import DEVICE, DISTILBERT_BASE, NUM_EPOCHS, DATASET_PATH, BATCH_SIZE, POSITIVE_SENTIMENT_THRESHOLD, NEUTRAL_SENTIMENT_THRESHOLD
from datasets import Dataset
from preprocess import load_csv
from processing import (
    tokenize,
    wordwise_sentiment_analysis,
)
from trainer import train_aspect_sentiment_extractor

logger = logging.getLogger(__name__)

def write_to_results_file(
    filename: str,
    true_tags: list,
    predicted_tags: list,
    label: str,
    mode: str = "w"
) -> None:
    """write accuracy and classification reports for model tags using true tags

    Args:
        filename (str): name of results file 
        true_tags (list): list of true tags
        predicted_tags (list): model predicted tags
        label (str): label for the results section
        mode (str, optional): file write mode. Defaults to "w".
    """
    with open(filename, mode, encoding="utf-8") as f:
        f.write(f"{label} Accuracy: {accuracy_score(true_tags, predicted_tags)}\n")
        f.write(f"{label} Classification report:\n {classification_report(true_tags, predicted_tags)}\n\n")

def prepare_aspect_dataset(
        sentences: list, 
        review_ids: list[str], 
        review_inputs: dict[str, torch.Tensor], 
        true_sentiments: list[int]
        ) -> tuple[Dataset, list[str], torch.Tensor, dict[str, int]]:
    # Sort aspects to have consistent indexing
    aspects = sorted(set([sentence.category for sentence in sentences]))
    aspect_to_idx = {aspect: idx for idx, aspect in enumerate(aspects)}

    review_id_to_idx = {review_id: idx for idx, review_id in enumerate(review_ids)}

    # Calculate aspect weights to handle class imbalance
    aspects_counter = Counter([sentence.category for sentence in sentences])
    total_sentences = len(sentences)
    aspect_weights = torch.tensor([total_sentences / (len(aspects) * aspects_counter[aspect]) for aspect in aspects], dtype=torch.float).to(DEVICE)
    
    # Make dataset on sentence by sentence basis
    tokenised_sentence_dataset = Dataset.from_dict({
        "input_ids": [review_inputs["input_ids"][review_id_to_idx[sentence.review.review_id]] for sentence in sentences],
        "attention_mask": [review_inputs["attention_mask"][review_id_to_idx[sentence.review.review_id]] for sentence in sentences],
        "aspect": [aspect_to_idx[sentence.category] for sentence in sentences],
        "sentiment": [true_sentiments[review_id_to_idx[sentence.review.review_id]] for sentence in sentences]
        })
    
    tokenised_sentence_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "aspect", "sentiment"],
    )
    
    return tokenised_sentence_dataset, aspects, aspect_weights, review_id_to_idx

def map_rating_to_sentiment(rating: float) -> int:
    """maps from rating to sentiment class

    Args:
        rating (float): The rating value to be mapped. In range [1, 10].

    Returns:
        int: The corresponding sentiment class (0: Negative, 1: Neutral, 2: Positive).
    """
    if rating >= POSITIVE_SENTIMENT_THRESHOLD:
        return 2  # Positive
    elif rating >= NEUTRAL_SENTIMENT_THRESHOLD:
        return 1  # Neutral
    else:
        return 0  # Negative

def main() -> None:
    args = get_args()
    results_path = f"results/{args.results}.txt"

    if args.model == "pre_fine_tuned_distilBERT":
        # Use pre-fine-tuned model from Hugging Face
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    elif args.model == "custom_fine_tuned_distilBERT":
        # Use model that has already been fine-tuned and saved
        model_name = "./models/fine_tuned_model"
    else:
        # Fine tune the base model
        model_name = DISTILBERT_BASE

    # Load dataset
    reviews, sentences = load_csv(DATASET_PATH)

    logger.info(f"Loaded {len(reviews)} reviews from the dataset.")
    logger.info(f"Loaded {len(sentences)} sentences from the dataset.")

    # Prepare true sentiments using combination of ratings and word-wise sentiment analysis
    combined_ratings = [review.rating + wordwise_sentiment_analysis(review) for review in reviews.values()]

    # Map combined ratings to sentiment classes: 0 (negative), 1 (neutral), 2 (positive)
    true_sentiments = [map_rating_to_sentiment(float(rating)) for rating in combined_ratings]

    review_ids = list(reviews.keys())

    # Tokenise reviews and get word embeddings
    review_inputs = tokenize([x.review for x in list(reviews.values())], DISTILBERT_BASE)
    logger.info(f"Tokenised {len(review_inputs['input_ids'])} texts.")

    if args.is_sentiment:
        # Performing sentiment analysis only
        # run_sentiment_analysis(model_name)
        pass
    else: 
        if not sentences:
            logger.error("No sentences found in dataset for aspect-based analysis.")
            exit(1)

        # Performing aspect extraction and sentiment analysis at sentence level
        logger.info("Performing aspect extraction...")

        tokenised_sentence_dataset, aspects, aspect_weights, review_id_to_idx = prepare_aspect_dataset(sentences=sentences, review_ids=review_ids, review_inputs=review_inputs, true_sentiments=true_sentiments)
        
        # Index of the sentence's review embedding in the word embeddings
        sentence_indices = [review_id_to_idx[sentence.review.review_id] for sentence in sentences]

        aspect_sentiment_extractor = AspectSentimentExtractor(model_name, num_aspects=len(aspects)).to(DEVICE)

        if not args.no_training: 
            logger.info("Training aspect sentiment extractor...")
            # Train output heads for aspect and sentiment classification
            train_aspect_sentiment_extractor(
                model=aspect_sentiment_extractor,
                dataset=tokenised_sentence_dataset,
                aspect_criterion=torch.nn.CrossEntropyLoss(weight=aspect_weights),
                sentiment_criterion=torch.nn.CrossEntropyLoss(),
                device=DEVICE,
                num_epochs=NUM_EPOCHS,
            )
            logger.info("Saving aspect sentiment extractor model...")
            torch.save(aspect_sentiment_extractor.state_dict(), "./models/aspect_sentiment_extractor_4_layers.pth")
        elif args.model == "aspect_sentiment_extractor":
            logger.info("Loading aspect sentiment extractor model...")
            aspect_sentiment_extractor.load_state_dict(torch.load("./models/aspect_sentiment_extractor.pth", map_location=DEVICE))

        logger.info("Running aspect and sentiment inference...")
        # Carry out inference with trained model
        aspect_predictions, sentiment_predictions = aspect_sentiment_extractor.aspect_sentiment_inference(
            input_ids=tokenised_sentence_dataset["input_ids"],
            attention_masks=tokenised_sentence_dataset["attention_mask"],
            device=DEVICE,
            batch_size=BATCH_SIZE
        )

        # Write results with accuracy and classification reports to file
        write_to_results_file(
            results_path,
            true_tags=tokenised_sentence_dataset['aspect'],
            predicted_tags=aspect_predictions,
            label="Aspect (Sentence-level)",
            mode="w"
        )
        write_to_results_file(
            results_path,
            true_tags=tokenised_sentence_dataset['sentiment'],
            predicted_tags=sentiment_predictions,
            label="Sentiment (Sentence-level)",
            mode="a"
        )

if __name__ == "__main__":
    main()