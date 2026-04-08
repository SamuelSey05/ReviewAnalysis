from google_play_scraper import Sort, reviews
import pandas as pd

from release_notes_scraping import normalize_text

def get_reviews(app_id, num_reviews=1000):

    result, continuation_token = reviews(
        app_id,
        lang='en', # English reviews
        country='us', # United States,
        sort=Sort.NEWEST, # Sort by newest reviews
        count=num_reviews,
    )

    df = pd.DataFrame(result)
    df["content"] = df["content"].fillna("").map(normalize_text)
    return df

if __name__ == "__main__":
    app_id = 'com.Slack'
    df = get_reviews(app_id, num_reviews=1000)
    columns_to_save = ["reviewId", "content", "score", "at", "replyContent", "repliedAt"]
    df.to_csv('datasets/slack_reviews.csv', columns=columns_to_save, index=False)