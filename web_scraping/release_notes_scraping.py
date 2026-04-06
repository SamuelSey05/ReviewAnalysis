import time
import csv
from ftfy import fix_text
from unidecode import unidecode
from playwright.sync_api import sync_playwright, Locator

def normalize_text(text: str) -> str:
    """Normalize text by fixing encoding issues and removing special characters.

    Args:
        text (str): Text to normalize.

    Returns:
        str: Normalized text.
    """
    try:
        text = text.encode().decode('unicode-escape')
    except:
        pass

    text = fix_text(text)
    return unidecode(text)


def extract_bug_fix_bullets(release: Locator, version: str, date: str) -> list[str]:
    """Extract bullet-point bug fixes from a release-note article.

    Args:
        release (Locator): Playwright locator for the release-note article.
        version (str): Version string to remove from bullet points.
        date (str): Date string to remove from bullet points.

    Returns:
        list[str]: List of cleaned bullet-point bug fixes from the release note.
    """

    bullets: list[str] = []
    bullet_items = release.locator("li")

    for index in range(bullet_items.count()):
        bullet_text = clean_release_text(bullet_items.nth(index).inner_text().strip(), version, date)
        if bullet_text:
            bullets.append(bullet_text)

    return bullets


def clean_release_text(text: str, version: str, date: str) -> str:
    """Normalize release-note text and strip shared labels.

    Args:
        text (str): Release text to clean
        version (str): Version string to remove from release text.
        date (str): Date string to remove from release text.

    Returns:
        str: Cleaned release text with version, date, and common labels removed.
    """

    return (
        normalize_text(text)
        .replace(version, "")
        .replace(date, "")
        .replace("Bug fixes", "")
        .replace("What's new", "")
        .strip()
    )

def scrape_release_notes(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            viewport={'width': 1280, 'height': 800}
        )

        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=60000)

        # Scroll down to load more release notes
        # for _ in range(5):
        #     page.keyboard.press("End")
        #     time.sleep(2)  

        try:
            page.get_by_role("button", name="ACCEPT ALL COOKIES").click(timeout=5000)
            print("Banner dismissed.")
        except Exception:
            print("Banner did not appear or was already closed.")

        time.sleep(5)
        
        releases = page.locator("div.release-note article").all()

        print(f"Found {len(releases)} release notes.")

        with open("datasets/slack_release_notes.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["version", "date", "content"])
            writer.writeheader()

            for release in releases:
                version = normalize_text(release.locator("h2").inner_text().strip())
                date = normalize_text(release.locator("p").first.inner_text().strip())

                snippets = extract_bug_fix_bullets(release, version, date)
                snippets = [snippet for snippet in snippets if snippet]

                if not snippets:
                    raw_content = clean_release_text(release.inner_text().strip(), version, date)
                    snippets = [raw_content] if raw_content else []

                for snippet in snippets:
                    writer.writerow({
                        "version": version.strip(),
                        "date": date.strip(),
                        "content": snippet.strip(),
                    })

        browser.close()    

if __name__ == "__main__":
    url = "https://slack.com/release-notes/android"
    scrape_release_notes(url)
    print("Release notes scraped and saved to slack_release_notes.csv")