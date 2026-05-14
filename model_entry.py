import argparse
import logging
import os
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
import torch

from src.config import BATCH_SIZE, DATASET_PATH, DEFAULT_MODEL_WEIGHTS_PATH, DEVICE, DISTILBERT_BASE, NUM_EPOCHS
from src.model_architecture import AspectSentimentExtractor
from src.processing import load_aspect_labels, load_dataset_csv, map_rating_to_sentiment, prepare_aspect_dataset, tokenize, wordwise_sentiment_analysis
from src.setup_dataset import verify_aware_dataset_is_present
from src.trainer import train_aspect_sentiment_extractor

logger = logging.getLogger(__name__)

def check_valid_file(path: str, accepted_extensions: list) -> None:
    """Checks that filepath inputted exists and has accepted extension.

    Args:
        path (str): Path to be checked.
        accepted_extensions (list): List of accepted file extensions

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the specified file does not have an accepted extension.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Specified file {path} does not exist.")

    if not any(path.endswith(ext) for ext in accepted_extensions):
        raise ValueError(f"Specified file {path} must have one of the following extensions: {', '.join(accepted_extensions)}.")
    
def get_correct_model_weights_path(model_name: str, provided_path: str | None) -> str:
    """Makes sure that the provided path for the model weight path is correct, or defaults to base ("./models/aspect_sentiment_extractor.pth") if needed

    Args:
        model_name (str): Model name to be used to base the model on
        provided_path (str | None): User provided path for model weights (optional)

    Returns:
        str: Path to be used for loading model weights
    """

    model_weights_path = DEFAULT_MODEL_WEIGHTS_PATH
    if provided_path:
        if model_name != DISTILBERT_BASE:
            raise ValueError("Cannot load custom weights when not using distilbert-base-uncased model. Please set --model to distilbert-base-uncased or remove --load_weights_from argument.")
        check_valid_file(provided_path, accepted_extensions=[".pth", ".pt"])
        model_weights_path = provided_path

    return model_weights_path

def write_accuracy_and_classification_to_results_file(
    filename: str,
    dataset: Dataset,
    aspect_predicted_tags: list,
    sentiment_predicted_tags: list,
) -> None:
    """Write accuracy and classification reports for model tags using true tags.

    Args:
        filename (str): Path to the file where results will be written.
        dataset (Dataset): Dataset containing true aspect and sentiment tags.
        aspect_predicted_tags (list): Predicted aspect tags from the model.
        sentiment_predicted_tags (list): Predicted sentiment tags from the model.
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Aspect (Sentence-level) Accuracy: {accuracy_score(dataset['aspect'], aspect_predicted_tags)}\n")
        f.write(f"Aspect (Sentence-level) Classification report:\n {classification_report(dataset['aspect'], aspect_predicted_tags)}\n\n")

        f.write(f"Sentiment (Sentence-level) Accuracy: {accuracy_score(dataset['sentiment'], sentiment_predicted_tags)}\n")
        f.write(f"Sentiment (Sentence-level) Classification report:\n {classification_report(dataset['sentiment'], sentiment_predicted_tags)}\n\n")

def main():
    """Main entrypoint for interaction with the AspectSentimentExtractor model
    """

    argparser = argparse.ArgumentParser(description="Entrypoint for interaction with the AspectSentimentExtractor model")
    argparser.add_argument(
        "--model", 
        type=str, 
        default=DISTILBERT_BASE, 
        help="Model name or path to load for aspect/sentiment extraction (default: distilbert-base-uncased)"
    )
    argparser.add_argument("--train", 
        action="store_true", 
        help="Whether to run training before inference (default: False)"
    )
    argparser.add_argument("--results", 
        type=str, 
        default="aspect_sentiment_results", 
        help="Filename (without extension) to write results to in the results/model_eval/ directory (default: aspect_sentiment_results)"
    )
    argparser.add_argument("--load_weights_from", 
        type=str, 
        help="Path to model weights to load for inference (can only use with distilbert-base-uncased model) (optional)"
    )
    argparser.add_argument("--single_review_file", 
        type=str, 
        help="Path to a text file containing a single review to run inference on (optional)"
    )
    
    args = argparser.parse_args()

    model_weights_path = get_correct_model_weights_path(args.model, args.load_weights_from)
    
    aspect_labels = load_aspect_labels()

    model = AspectSentimentExtractor(args.model, num_aspects=len(aspect_labels)).to(DEVICE)

    verify_aware_dataset_is_present(DATASET_PATH)

    reviews, opinions = load_dataset_csv(DATASET_PATH)

    # Prepare true sentiments using combination of ratings and word-wise sentiment analysis
    combined_ratings = [review.rating + wordwise_sentiment_analysis(review) for review in reviews.values()]

    # Map combined ratings to sentiment classes: 0 (negative), 1 (neutral), 2 (positive)
    true_sentiments = [map_rating_to_sentiment(float(rating)) for rating in combined_ratings]

    review_ids = list(reviews.keys())

    # Tokenise reviews and get word embeddings
    review_inputs = tokenize([x.review for x in list(reviews.values())], DISTILBERT_BASE)
    logger.info(f"Tokenised {len(review_inputs['input_ids'])} texts.")

    dataset, _, aspect_weights, _ = prepare_aspect_dataset(opinions, review_ids, review_inputs, true_sentiments)

    if args.train:
        logger.info("Training aspect sentiment extractor...")

        train_aspect_sentiment_extractor(
            model=model,
            dataset=dataset,
            aspect_criterion=torch.nn.CrossEntropyLoss(weight=aspect_weights),
            sentiment_criterion=torch.nn.CrossEntropyLoss(),
            num_epochs=NUM_EPOCHS,
            opinions=opinions,
            plot_progress=True,
        )
        
        logger.info("Saving aspect sentiment extractor model...")
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), "./models/aspect_sentiment_extractor.pth")
        logger.info("Model saved to ./models/aspect_sentiment_extractor.pth")
    else:
        # Load the model for inference
        logger.info("Loading aspect sentiment extractor model...")
        try:
            model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights for inference: {e}.")


    # Inference
    model.eval()

    if not args.single_review_file:
        # Inference on dataset, with comparison to true tags
        logger.info("Running aspect and sentiment inference on dataset...")

        aspect_predictions, sentiment_predictions = model.aspect_sentiment_inference(list(map(lambda x: x.review.review, opinions)), batch_size=BATCH_SIZE)

        # Write results with accuracy and classification reports to file
        results_path = f"./results/model_eval/{args.results}.txt" if args.results else "./results/model_eval/aspect_sentiment_results.txt"

        write_accuracy_and_classification_to_results_file(
            results_path,
            dataset=dataset,
            aspect_predicted_tags=aspect_predictions,
            sentiment_predicted_tags=sentiment_predictions,
        )

        cm = confusion_matrix(dataset['aspect'], aspect_predictions, normalize='true')

        # Plotting the matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=aspect_labels)

        # Use a colormap like 'Blues' or 'Greys' to match academic standards
        disp.plot(cmap=plt.get_cmap('Greys'), ax=ax, xticks_rotation=45)

        plt.title("Confusion Matrix for Aspect Classification (DistilBERT)")
        plt.tight_layout()
        plt.savefig("./results/model_eval/aspect_classification_confusion_matrix.png")

        logger.info(f"Results written to {results_path}")
    else:
        # Inference on single review text file
        logger.info(f"Running aspect and sentiment inference on single review file: {args.single_review_file}...")

        check_valid_file(args.single_review_file, accepted_extensions=[".txt"])

        with open(args.single_review_file, "r", encoding="utf-8") as f:
            single_review =  f.read().strip()

        aspect_predictions, sentiment_predictions = model.aspect_sentiment_inference([single_review], batch_size=1)

        sentiment_labels = ["negative", "neutral", "positive"]

        print(f"Predicted aspect: {aspect_labels[int(aspect_predictions[0])]} (Class {int(aspect_predictions[0])})")
        print(f"Predicted sentiment: {sentiment_labels[int(sentiment_predictions[0])]} (Class {int(sentiment_predictions[0])})")


if __name__ == "__main__":
    main()