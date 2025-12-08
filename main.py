import argparse

from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from preprocess import load_csv
from processing import tokenize, inference

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    args = argparser.parse_args()

    reviews, review_parts = load_csv("datasets/AWARE_Comprehensive.csv")

    print(f"Loaded {len(reviews)} reviews from the dataset.")
    print(f"Loaded {len(review_parts)} review parts from the dataset.")

    inputs = tokenize(reviews, "distilbert-base-uncased-finetuned-sst-2-english")

    print(f"Tokenised {len(inputs)} texts.")

    predictions = []
    true_labels = []

    for review, tokens in tqdm(inputs.items(), desc="Running inference"):
        predictions.append(inference(tokens, "distilbert-base-uncased-finetuned-sst-2-english").item())
        true_labels.append([1 if review.rating >= 4 else 0])

    print(predictions)

    print(accuracy_score(true_labels, predictions))
    print(classification_report(true_labels, predictions))

