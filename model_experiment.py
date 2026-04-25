import csv
import os
import torch

from aspect_based import AspectSentimentExtractor
from config import DEVICE, DISTILBERT_BASE
from release_notes import release_notes_vs_reviews_comparison


def write_results_to_csv(sorted_results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "release_note_id",
                "review_id",
                "similarity",
                "lcs_length",
                "time_diff_days",
                "release_note",
                "review",
            ],
        )
        writer.writeheader()

        for (release_note_id, review_id), result in sorted_results:
            writer.writerow({
                "release_note_id": release_note_id,
                "review_id": review_id,
                "similarity": result.similarity,
                "lcs_length": result.lcs_length,
                "time_diff_days": result.time_diff_days,
                "release_note": result.release_note.content,
                "review": result.review.content,
            })


def main():
    model = AspectSentimentExtractor(DISTILBERT_BASE, num_aspects=12).to(DEVICE)
    model.load_state_dict(torch.load("./models/aspect_sentiment_extractor.pth", map_location=DEVICE))
    model.eval()


    sorted_results, total_candidate_pairs, no_of_negative_reviews = release_notes_vs_reviews_comparison(
            model,
            "zoom",
            output_file="./experiment/release_notes_comparison_results.json",
        )

    write_results_to_csv(sorted_results, "./experiment/zoom_filtered_results.csv")

    

if __name__ == "__main__":
    main()