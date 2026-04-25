import pandas as pd
import os

def generate_experiment_csvs():
    # 1. Load Datasets
    reviews_df = pd.read_csv('datasets/slack_reviews.csv')
    notes_df = pd.read_csv('datasets/slack_release_notes.csv')

    # 2. Convert dates for filtering logic
    reviews_df['at'] = pd.to_datetime(reviews_df['at'])
    # Release note dates in Slack CSV use format like '11 March 2026'
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