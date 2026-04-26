"""Fetch latest Church News headlines and save for Rosie."""
import re, os, json, urllib.request, html
from datetime import datetime

OUT_FILE = "church/news.json"
URL = "https://www.thechurchnews.com/"

def fetch_news():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    # Extract headlines from link text — look for article links with titles
    headlines = []
    seen = set()
    # Match links with descriptive text (skip nav/footer links)
    for match in re.finditer(r'\[([^\]]{20,200})\]\(\s*(https://www\.thechurchnews\.com/[^)]+)\s*\)', raw):
        title = match.group(1).strip()
        url = match.group(2).strip()
        # Skip duplicates and non-article links
        if title in seen or "/authors/" in url or "/pages/" in url:
            continue
        # Skip navigation items
        if title.lower() in {"church news", "deseret news", "subscribe", "sign in", "register"}:
            continue
        seen.add(title)
        headlines.append({"title": title, "url": url})

    if not headlines:
        # Fallback: extract from HTML anchor tags
        for match in re.finditer(r'<a[^>]+href="(https://www\.thechurchnews\.com/[^"]+)"[^>]*>([^<]{20,200})</a>', raw):
            url = match.group(1)
            title = html.unescape(match.group(2)).strip()
            if title not in seen and "/authors/" not in url:
                seen.add(title)
                headlines.append({"title": title, "url": url})

    result = {
        "fetched": datetime.now().isoformat(),
        "headlines": headlines[:15]  # Keep top 15
    }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {len(result['headlines'])} headlines to {OUT_FILE}")
    for h in result["headlines"]:
        print(f"  - {h['title']}")

if __name__ == "__main__":
    fetch_news()
