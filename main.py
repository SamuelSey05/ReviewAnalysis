from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset

from preprocess import load_csv
from processing import tokenize, inference, wordwise_sentiment_analysis
from fine_tuning import fine_tune_model
from arguements import get_args

if __name__ == "__main__":
    args = get_args()

    if args.model == "pre_fine_tuned_distilBERT":
        # Use pre-fine-tuned model from Hugging Face
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    elif args.model == "custom_fine_tuned_distilBERT":
        # Use model that has already been fine-tuned and saved
        model_name = "./models/fine_tuned_model"
    else:
        # Fine tune the base model
        model_name = "distilbert-base-uncased"

    reviews, sentences = load_csv("datasets/AWARE_Comprehensive.csv")

    print(f"Loaded {len(reviews)} reviews from the dataset.")
    print(f"Loaded {len(sentences)} sentences from the dataset.")

    combined_ratings = [review.rating + wordwise_sentiment_analysis(review) for review in reviews]

    true_labels = [2 if combined_ratings[i] >= 7.5 else (1 if combined_ratings[i] >= 4 else 0) for i in range(len(combined_ratings))]

    inputs = tokenize(reviews, model_name)
    print(f"Tokenised {len(inputs)} texts.")

    tokenised_dataset = Dataset.from_dict({"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "label": true_labels})

    if model_name == "distilbert-base-uncased":
        print("Fine tuning model")
        model = fine_tune_model(tokenised_dataset, model_name)
        model_name = "./models/fine_tuned_model"

    predictions = [inference({key: inputs[key][i] for key in inputs}, model_name).item() for i in tqdm(range(len(inputs["input_ids"])), desc="Running inference")]

    with open(f"results/{args.results}.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy_score(true_labels, predictions)}\n")
        f.write(f"Classification report:\n {classification_report(true_labels, predictions)}\n")

