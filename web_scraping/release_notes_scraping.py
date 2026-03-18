import time
import csv
from playwright.sync_api import sync_playwright

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
                version = release.locator("h2").inner_text().strip()

                date = release.locator("p").first.inner_text().strip()
                content = release.inner_text().strip().replace(version, "").replace(date, "").replace("\n", "").replace("Bug fixes", "").replace("What's new", "").lower().strip()

                writer.writerow({
                    "version": version.strip(),
                    "date": date.strip(),
                    "content": content.strip()
                })

        browser.close()    

if __name__ == "__main__":
    url = "https://slack.com/release-notes/android"
    scrape_release_notes(url)
    print("Release notes scraped and saved to slack_release_notes.csv")