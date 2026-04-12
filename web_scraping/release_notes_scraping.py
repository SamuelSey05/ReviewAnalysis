import time
import csv
import re
from ftfy import fix_text
from unidecode import unidecode
from playwright.sync_api import sync_playwright, Locator

RELEASE_NOTE_FIELDS = ["release_note_id", "version", "date", "content"]

def normalise_text(text: str) -> str:
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
        normalise_text(text)
        .replace(version, "")
        .replace(date, "")
        .replace("Bug fixes", "")
        .replace("What's new", "")
        .strip()
    )


def extract_date_from_discord_url(article_url: str) -> str:
    """Extract release date from a Discord release note article URL.

    Args:
        article_url (str): URL of the Discord release note article to extract the date from.

    Returns:
        str: String representation of the release date in form "DD Month YYYY".
    """

    match = re.search(r"discord-patch-notes-([a-z]+)(?:-(\d{1,2}))?-(\d{4})/?$", article_url)
    if not match:
        return ""

    month, day, year = match.groups()
    # Fallback on 30th if day is not present in URL
    day = day or "30"
    return f"{int(day)} {month.capitalize()} {year}"


def write_release_note_row(writer: csv.DictWriter, release_note_id: str, version: str, date: str, content: str) -> None:
    """Write row to csv for a release note.

    Args:
        writer (csv.DictWriter): CSV DictWriter object to write the row with.
        release_note_id (str): ID for the release note, should be unique and stable across runs.
        version (str): Version string for the release note, e.g. "Discord 30 September 2023".
        date (str): Release date string, e.g. "30 September 2023".
        content (str): Content of the release note.
    """

    writer.writerow({
        "release_note_id": release_note_id,
        "version": version.strip(),
        "date": date.strip(),
        "content": content.strip(),
    })

def scrape_release_notes_discord(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("a[href*='/blog/discord-patch-notes-']", timeout=15000)

        article_locators = page.locator("a.featured_main-card, a.cms_article").all()
        articles = []
        for locator in article_locators:
            href = locator.get_attribute("href")
            if href :
                articles.append(f"https://discord.com{href}" if href.startswith("/") else href)

        print(f"Found {len(articles)} release note articles.")

        with open("datasets/discord_release_notes.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=RELEASE_NOTE_FIELDS)
            writer.writeheader()
            release_note_id = 0

            for article_url in articles:

                print(f"Scraping article: {article_url}")

                parsed_date = extract_date_from_discord_url(article_url)

                page.goto(article_url, wait_until="networkidle", timeout=60000)

                page.wait_for_selector("article.article_rich-text-2", timeout=15000)           

                container = page.locator("article.article_rich-text-2")

                elements = container.locator("> *").all()

                current_category = "General"
                for el in elements:
                    tag_name = el.evaluate("node => node.tagName").lower()
                    if tag_name == "h2":
                        current_category = el.inner_text().strip()
                    elif tag_name == "ul":
                        bullet_items = el.locator("li").all()
                        for index in range(len(bullet_items)):
                            bullet_text = clean_release_text(bullet_items[index].inner_text().strip(), "", "")
                            if bullet_text:
                                release_note_id += 1
                                write_release_note_row(
                                    writer,
                                    f"discord-{release_note_id}",
                                    f"Discord {parsed_date}",
                                    parsed_date,
                                    f"{current_category.strip()}: {bullet_text.strip()}",
                                )


def scrape_release_notes(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
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
            writer = csv.DictWriter(file, fieldnames=RELEASE_NOTE_FIELDS)
            writer.writeheader()
            release_note_id = 0

            for release in releases:
                version = normalise_text(release.locator("h2").inner_text().strip())
                date = normalise_text(release.locator("p").first.inner_text().strip())

                snippets = extract_bug_fix_bullets(release, version, date)
                snippets = [snippet for snippet in snippets if snippet]

                if not snippets:
                    raw_content = clean_release_text(release.inner_text().strip(), version, date)
                    snippets = [raw_content] if raw_content else []

                for snippet in snippets:
                    release_note_id += 1
                    write_release_note_row(
                        writer,
                        f"slack-{release_note_id}",
                        version,
                        date,
                        snippet,
                    )

        browser.close()    

if __name__ == "__main__":
    url = "https://slack.com/release-notes/android"
    scrape_release_notes(url)
    print("Release notes scraped and saved to slack_release_notes.csv")

    url = "https://discord.com/tags/patch-notes"
    scrape_release_notes_discord(url)
    print("Release notes scraped and saved to discord_release_notes.csv")