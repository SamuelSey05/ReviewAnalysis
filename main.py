import argparse

from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset

from preprocess import load_csv
from processing import tokenize, inference
from fine_tuning import fine_tune_model

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "model",
        type=str,
        help="Name of the pre-trained model to use for inference."
    )

    args = argparser.parse_args()

    if args.model == "pre_fine_tuned_distilBERT":
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    else:
        model_name = "distilbert-base-uncased"

    reviews, sentences = load_csv("datasets/AWARE_Comprehensive.csv")

    print(f"Loaded {len(reviews)} reviews from the dataset.")
    print(f"Loaded {len(sentences)} sentences from the dataset.")

    true_labels = [1 if review.rating >= 4 else 0 for review in reviews]

    inputs = tokenize(reviews, model_name)
    print(f"Tokenised {len(inputs)} texts.")

    tokenised_dataset = Dataset.from_dict({"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "label": true_labels})

    if model_name == "distilbert-base-uncased":
        print("Fine tuning model")
        model = fine_tune_model(tokenised_dataset, model_name)
        model_name = "./models/fine_tuned_model"

    predictions = []

    for i in tqdm(range(len(inputs["input_ids"])), desc="Running inference"):
        predictions.append(inference({key: inputs[key][i] for key in inputs}, model_name).item())

    with open("results/pre_fine_tuned_distilBERT.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy_score(true_labels, predictions)}\n")
        f.write(f"Classification report: {classification_report(true_labels, predictions)}\n")

