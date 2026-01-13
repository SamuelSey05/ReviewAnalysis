from typing import Counter

import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from arguements import get_args
from aspect_based import AspectExtractor
from datasets import Dataset
from fine_tuning import fine_tune_model
from preprocess import load_csv
from processing import (
    get_word_embeddings,
    sentiment_inference,
    tokenize,
    wordwise_sentiment_analysis,
)
from trainer import train_aspect_extractor

DISTILBERT_BASE = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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
        model_name = DISTILBERT_BASE

    reviews, sentences = load_csv("datasets/AWARE_Comprehensive.csv")

    print(f"Loaded {len(reviews)} reviews from the dataset.")
    print(f"Loaded {len(sentences)} sentences from the dataset.")

    combined_ratings = [review.rating + wordwise_sentiment_analysis(review) for review in reviews.values()]

    true_sentiments = [2 if combined_ratings[i] >= 7.5 else (1 if combined_ratings[i] >= 4 else 0) for i in range(len(combined_ratings))]

    review_ids = list(reviews.keys())

    review_inputs = tokenize(list(reviews.values()), DISTILBERT_BASE)
    print(f"Tokenised {len(review_inputs['input_ids'])} texts.")

    word_embeddings = get_word_embeddings(review_inputs, model_name, DEVICE)

    if args.is_sentiment:
        tokenised_review_dataset = Dataset.from_dict({"input_ids": list(review_inputs["input_ids"]), "attention_mask": list(review_inputs["attention_mask"]), "sentiment": true_sentiments})

        if model_name == DISTILBERT_BASE:
            print("Fine tuning model")
            model = fine_tune_model(tokenised_review_dataset, DISTILBERT_BASE, optimise_hyperparameters=args.optimize_hyperparameters)
            model_name = "./models/fine_tuned_model"
        print("Performing sentiment analysis...")
        predictions = sentiment_inference(word_embeddings, model_name, DEVICE).tolist()

        with open(f"results/{args.results}.txt", "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {accuracy_score(true_sentiments, predictions)}\n")
            f.write(f"Classification report:\n {classification_report(true_sentiments, predictions)}\n")
    else: 
        print("Performing aspect extraction...")

        aspects = sorted(set([sentence.category for sentence in sentences]))
        aspect_to_idx = {aspect: idx for idx, aspect in enumerate(aspects)}
        review_id_to_idx = {review_id: idx for idx, review_id in enumerate(review_ids)}

        aspects_counter = Counter([sentence.category for sentence in sentences])
        total_sentences = len(sentences)

        aspect_weights = torch.tensor([total_sentences / (len(aspects) * aspects_counter[aspect]) for aspect in aspects], dtype=torch.float).to(DEVICE)
        
        tokenised_sentence_dataset = Dataset.from_dict({
            "input_ids": [review_inputs["input_ids"][review_id_to_idx[sentence.review.review_id]] for sentence in sentences],
            "attention_mask": [review_inputs["attention_mask"][review_id_to_idx[sentence.review.review_id]] for sentence in sentences],
            "aspect": [aspect_to_idx[sentence.category] for sentence in sentences],
            "sentiment": [true_sentiments[review_id_to_idx[sentence.review.review_id]] for sentence in sentences]
            })
        
        # Index of the sentence's review embedding in the word embeddings
        sentence_indices = [review_id_to_idx[sentence.review.review_id] for sentence in sentences]

        aspect_extractor = AspectExtractor(model_name, num_aspects=len(aspects)).to(DEVICE)

        train_aspect_extractor(
            model=aspect_extractor,
            dataset=tokenised_sentence_dataset,
            embeddings=word_embeddings,
            sentence_indices=sentence_indices,
            optimiser=torch.optim.AdamW(aspect_extractor.parameters(), lr=5e-5),
            aspect_criterion=torch.nn.CrossEntropyLoss(weight=aspect_weights),
            sentiment_criterion=torch.nn.CrossEntropyLoss(),
            device=DEVICE,
            num_epochs=20,
        )

        aspect_extractor.eval()
        aspect_predictions = []
        sentiment_predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sentence_indices), 64), desc="Running inference"):
                batch_indices = torch.tensor(sentence_indices[i:i+64], dtype=torch.long)
                batch_embeddings = word_embeddings[batch_indices].to(DEVICE)
                batch_attention_mask = torch.tensor(tokenised_sentence_dataset["attention_mask"][i:i+64]).to(DEVICE)
                aspect_logits, sentiment_logits = aspect_extractor.forward(batch_embeddings, batch_attention_mask)
                aspect_predictions.extend(torch.argmax(aspect_logits, dim=-1).cpu().tolist())
                sentiment_predictions.extend(torch.argmax(sentiment_logits, dim=-1).cpu().tolist())

        with open(f"results/{args.results}.txt", "w", encoding="utf-8") as f:
            f.write(f"Aspect Accuracy: {accuracy_score(tokenised_sentence_dataset['aspect'], aspect_predictions)}\n")
            f.write(f"Aspect Classification report:\n {classification_report(tokenised_sentence_dataset['aspect'], aspect_predictions)}\n\n")
            f.write(f"Sentiment Accuracy: {accuracy_score(tokenised_sentence_dataset['sentiment'], sentiment_predictions)}\n")
            f.write(f"Sentiment Classification report:\n {classification_report(tokenised_sentence_dataset['sentiment'], sentiment_predictions)}\n\n")
