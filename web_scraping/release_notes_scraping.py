import json
import time
import csv
import re
from datetime import datetime
from ftfy import fix_text
from unidecode import unidecode
from playwright.sync_api import sync_playwright, Locator
from bs4 import BeautifulSoup

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
    bullets = []
    for item in release.locator("li").all():
        text = clean_release_text(item.inner_text().strip(), version, date)
        if text:
            bullets.append(text)
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
    text = normalise_text(text)
    for remove in [version, date, "Bug fixes", "What's new"]:
        text = text.replace(remove, "")
    return text.strip()


def normalise_release_date(date_text: str) -> str:
    """Covert date formats into DD Month YYYY

    Args:
        date_text (str): Date text to normalise

    Returns:
        str: Normalised date string
    """

    date_text = normalise_text(date_text).strip()
    for fmt in ["%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"]:
        try:
            parsed = datetime.strptime(date_text, fmt)
            return f"{parsed.day} {parsed.strftime('%B %Y')}"
        except ValueError:
            continue

    return date_text


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


def write_release_note_rows(output_path: str, app_prefix: str, rows: list[tuple[str, str, str]]) -> None:

    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RELEASE_NOTE_FIELDS)
        writer.writeheader()

        for idx, (version, date, content) in enumerate(rows, start=1):
            write_release_note_row(writer, f"{app_prefix}-{idx}", version, date, content)

def scrape_discord():
    """Scrape Discord release notes."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        rows = []

        page.goto("https://discord.com/tags/patch-notes", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("a[href*='/blog/discord-patch-notes-']", timeout=15000)

        articles = []
        for locator in page.locator("a.featured_main-card, a.cms_article").all():
            href = locator.get_attribute("href")
            if href:
                articles.append(f"https://discord.com{href}" if href.startswith("/") else href)

        print(f"Found {len(articles)} release note articles.")

        for article_url in articles:
            parsed_date = extract_date_from_discord_url(article_url)
            page.goto(article_url, wait_until="networkidle", timeout=60000)
            page.wait_for_selector("article.article_rich-text-2", timeout=15000)

            category = "General"
            for el in page.locator("article.article_rich-text-2 > *").all():
                tag = el.evaluate("node => node.tagName").lower()

                if tag == "h2":
                    category = el.inner_text().strip()
                elif tag == "ul":
                    for item in el.locator("li").all():
                        bullet = clean_release_text(item.inner_text().strip(), "", "")
                        if bullet:
                            rows.append((f"Discord {parsed_date}", parsed_date, f"{category}: {bullet}"))

        browser.close()

    write_release_note_rows("datasets/discord_release_notes.csv", "discord", rows)


def scrape_slack():
    """Scrape Slack release notes."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        rows = []

        page.goto("https://slack.com/release-notes/android", wait_until="networkidle", timeout=60000)

        try:
            page.get_by_role("button", name="ACCEPT ALL COOKIES").click(timeout=5000)
            print("Banner dismissed.")
        except Exception:
            print("Banner did not appear or was already closed.")

        time.sleep(5)
        
        releases = page.locator("div.release-note article").all()
        print(f"Found {len(releases)} release notes.")

        for release in releases:
            version = normalise_text(release.locator("h2").inner_text().strip())
            date = normalise_release_date(release.locator("p").first.inner_text().strip())

            snippets = [s for s in extract_bug_fix_bullets(release, version, date) if s]
            if not snippets:
                raw = clean_release_text(release.inner_text().strip(), version, date)
                if raw:
                    snippets = [raw]

            for snippet in snippets:
                rows.append((version, date, snippet))

        browser.close()

    write_release_note_rows("datasets/slack_release_notes.csv", "slack", rows)


def scrape_zoom():
    """Scrape Zoom release notes."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        rows = []

        page.goto("https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0061222", wait_until="domcontentloaded", timeout=60000)

        json_content = page.locator("script[type='application/ld+json']").inner_text()
        data = json.loads(json_content)
        soup = BeautifulSoup(data.get("articleBody", ""), "html.parser")

        version = ""
        date = ""

        for el in soup.find_all(["h3", "h4", "table", "p"]):
            text = el.get_text().strip()

            if el.name == "h3":
                date = normalise_release_date(text)
            elif el.name == "table" and any("Android" in th.get_text() for th in el.find_all("th")):
                cells = el.find_all("td")
                if len(cells) > 3:
                    version = cells[3].get_text().strip()
            elif el.name == "h4" and text == "Resolved issues":
                table = el.find_next_sibling("table") or el.find_next("table")
                if table:
                    for row in table.find_all("tr"):
                        cols = row.find_all("td")
                        if len(cols) > 1:
                            desc = cols[0].get_text().strip()
                            platforms = cols[1].get_text().strip()
                            if "Android" in platforms and desc:
                                rows.append((f"Zoom {version}", date, desc))

        browser.close()

    write_release_note_rows("datasets/zoom_release_notes.csv", "zoom", rows)


if __name__ == "__main__":
    scrape_slack()
    print("Release notes scraped and saved to slack_release_notes.csv")

    scrape_discord()
    print("Release notes scraped and saved to discord_release_notes.csv")

    scrape_zoom()
    print("Release notes scraped and saved to zoom_release_notes.csv")