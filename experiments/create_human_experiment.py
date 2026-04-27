import pandas as pd
import os

from src.config import DEVICE, DISTILBERT_BASE
from src.model_architecture import AspectSentimentExtractor
from src.release_notes_comparison_utils import release_notes_vs_reviews_comparison

def generate_experiment_csvs():
    # Load Datasets
    reviews_df = pd.read_csv('datasets/slack_reviews.csv')
    notes_df = pd.read_csv('datasets/slack_release_notes.csv')

    model = AspectSentimentExtractor(DISTILBERT_BASE, num_aspects=12).to(DEVICE)

    sorted_results, _, _ = release_notes_vs_reviews_comparison(model=model, app_name="slack")


    reviewID_to_matched_noteID = {}
    for _, result in sorted_results:
        if result.similarity >= 0.5 and result.review.review_id not in reviewID_to_matched_noteID:
            reviewID_to_matched_noteID[result.review.review_id] = result.release_note.release_note_id

    # Slack dates
    reviews_df['at'] = pd.to_datetime(reviews_df['at'])
    notes_df['date_dt'] = pd.to_datetime(notes_df['date'], format='%d %B %Y', errors='coerce')

    review_start, review_end = '2022-03-01', '2022-03-31'
    fix_end = '2022-04-30'

    # 3. Filter for the Experiment Window (August 2024)
    # Focus: Unfulfilled requirements (score <= 3)
    experiment_reviews = reviews_df[
        (reviews_df['at'] >= review_start) & 
        (reviews_df['at'] <= review_end) & 
        (reviews_df['score'] <= 3)
    ].copy()

    # Potential Fulfillments: August through October 2024
    experiment_notes = notes_df[
        (notes_df['date_dt'] >= review_start) & 
        (notes_df['date_dt'] <= fix_end)
    ].copy()

    # 4. Prepare Output directory
    os.makedirs('experiment', exist_ok=True)

    # 5. Create Manual Evaluation Worksheet
    # This is where you will record your manual findings and the time spent
    manual_eval = experiment_reviews[['reviewId', 'at', 'content']].copy()
    manual_eval['Human_Matched_Note_ID'] = ""
    manual_eval['Reasoning_Comments'] = ""

    for row in manual_eval.itertuples():
        manual_eval['Model_Matched_Note_ID'] = reviewID_to_matched_noteID.get(row.reviewId, "N/A")

    manual_eval.to_csv('experiment/manual_evaluation_experiment.csv', index=False)

    # 6. Create Release Note Reference Sheet
    # Keep this open to find IDs for the 'Human_Matched_Note_ID' column
    notes_reference = experiment_notes[['release_note_id', 'date', 'content']].copy()
    notes_reference.to_csv('experiment/experiment_release_notes_reference.csv', index=False)

    print("Successfully generated files in 'exeperiment/'")
    print(f"Total Reviews to process: {len(manual_eval)}")
    print(f"Search space (Release Notes): {len(notes_reference)}")

if __name__ == "__main__":
    generate_experiment_csvs()