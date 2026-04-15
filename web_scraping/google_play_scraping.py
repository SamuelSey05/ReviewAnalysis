import math
import logging

from google_play_scraper import Sort, reviews, app
import pandas as pd
from pathlib import Path

from release_notes_scraping import normalise_text

REVIEW_COLUMNS = ["reviewId", "content", "score", "at", "replyContent", "repliedAt"]
logger = logging.getLogger(__name__)

def get_reviews(app_id: str, num_reviews: int = 10000, page_size: int = 1000, per_page_keep: int = 100) -> pd.DataFrame:
    """Fetch and normalize Google Play reviews for one app.

    Args:
        app_id (str): Google Play application id.
        num_reviews (int): Approximate maximum number of reviews to collect.
        page_size (int): Number of reviews requested per API page.
        per_page_keep (int): Legacy parameter kept for compatibility; sampling is
            currently based on `num_reviews / total_reviews`.

    Returns:
        pd.DataFrame: Deduplicated review rows with REVIEW_COLUMNS present.
    """

    all_results = []
    continuation_token = None

    result = app(app_id, lang='en', country='us')
    total_reviews = result.get("reviews", 0)
    logger.info("Total reviews available for %s: %d", app_id, total_reviews)

    sample_rate = min(1.0, num_reviews / total_reviews) if total_reviews > 0 else 1.0

    while len(all_results) < num_reviews:
        batch_count = page_size
        if continuation_token is None:
            result, continuation_token = reviews(
                app_id,
                lang='en', # English reviews
                country='us', # United States,
                sort=Sort.NEWEST, # Sort by newest reviews
                count=batch_count,
            )
        else:
            result, continuation_token = reviews(
                app_id,
                lang='en', # English reviews
                country='us', # United States,
                sort=Sort.NEWEST, # Sort by newest reviews
                count=batch_count,
                continuation_token=continuation_token,
            )

        if not result:
            break

        # Keep only the first chunk of each page to push coverage further back in time.
        all_results.extend(result[:math.floor(len(result) * sample_rate)])

        if continuation_token is None:
            break

        logger.info("Collected %d reviews so far", len(all_results))

    df = pd.DataFrame(all_results)

    # Remove duplicates in case pagination returns overlapping rows.
    if "reviewId" in df.columns:
        df = df.drop_duplicates(subset=["reviewId"])

    for column in REVIEW_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df["content"] = df["content"].fillna("").map(normalise_text)
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    apps_df = pd.read_csv("web_scraping/apps.csv")

    for _, app_row in apps_df.iterrows():
        app_name = str(app_row["name"]).strip()
        app_id = str(app_row["app_id"]).strip()
        if not app_id:
            continue

        logger.info("Fetching reviews for %s (%s)...", app_name, app_id)
        df = get_reviews(app_id, num_reviews=10000, page_size=1000, per_page_keep=100)

        output_file = Path("datasets") / f"{app_name.lower()}_reviews.csv"
        df.to_csv(output_file, columns=REVIEW_COLUMNS, index=False)
        logger.info("Saved %d reviews to %s", len(df), output_file)