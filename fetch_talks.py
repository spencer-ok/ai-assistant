"""Fetch April 2026 General Conference talks and save as text files."""
import re, os, time, json, urllib.request, html

TALKS_DIR = "church/talks"
os.makedirs(TALKS_DIR, exist_ok=True)

# Talk URLs from churchofjesuschrist.org (April 2026)
TALKS = [
    ("oaks-alive-in-christ", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/49oaks?lang=eng"),
    ("eyring-prayers-for-peace", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/18eyring?lang=eng"),
    ("uchtdorf-encounter-empty-tomb", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/31uchtdorf?lang=eng"),
    ("freeman-best-days-worst-days", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/32freeman?lang=eng"),
    ("christofferson-character-of-christ", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/41christofferson?lang=eng"),
    ("kearon-about-his-business", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/13kearon?lang=eng"),
    ("bednar-endured-valiantly", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/16bednar?lang=eng"),
    ("rasband-he-is-risen", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/36rasband?lang=eng"),
    ("renlund-because-of-jesus-christ", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/37renlund?lang=eng"),
    ("andersen-eternal-marriage", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/45andersen?lang=eng"),
    ("stevenson-lost-luggage", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/22stevenson?lang=eng"),
    ("causse-love-all-love-each", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/25causse?lang=eng"),
    ("cook-keys-covenants-easter", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/46cook?lang=eng"),
    ("gong-eastertide", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/48gong?lang=eng"),
    ("soares-true-vine", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/28soares?lang=eng"),
    ("yee-ministering", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/14yee?lang=eng"),
    ("gilbert-come-home", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/15gilbert?lang=eng"),
    ("teh-follow-the-prophet", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/17teh?lang=eng"),
    ("becerra-tithing", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/19becerra?lang=eng"),
    ("larreal-saviors-love", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/33larreal?lang=eng"),
    ("rowe-christ-guide", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/34rowe?lang=eng"),
    ("mutombo-covenant-relationship", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/38mutombo?lang=eng"),
    ("walker-peculiar-treasure", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/39walker?lang=eng"),
    ("ortega-author-finisher", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/23ortega?lang=eng"),
    ("wu-give-away-sins", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/24wu?lang=eng"),
    ("wunderli-not-our-burden", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/26wunderli?lang=eng"),
    ("holmes-jesus-is-the-way", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/27holmes?lang=eng"),
    ("matswagothata-knows-by-name", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/29matswagothata?lang=eng"),
    ("wong-remember", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/42wong?lang=eng"),
    ("hall-glory-in-jesus", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/43hall?lang=eng"),
    ("porter-here-am-i", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/44porter?lang=eng"),
    ("wakolo-come-unto-christ", "https://www.churchofjesuschrist.org/study/general-conference/2026/04/47wakolo?lang=eng"),
]

def clean_html(raw):
    """Strip HTML tags and clean up text."""
    text = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', raw, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_talk(raw_html):
    """Extract the talk body from the page."""
    # Look for the main article content
    match = re.search(r'<article[^>]*>(.*?)</article>', raw_html, re.DOTALL)
    if match:
        return clean_html(match.group(1))
    # Fallback: look for body-block
    match = re.search(r'class="body-block"[^>]*>(.*?)</div>\s*</div>\s*</div>', raw_html, re.DOTALL)
    if match:
        return clean_html(match.group(1))
    return None

for name, url in TALKS:
    outpath = os.path.join(TALKS_DIR, f"{name}.txt")
    if os.path.exists(outpath) and os.path.getsize(outpath) > 500:
        print(f"  Skip {name} (exists)")
        continue
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        talk = extract_talk(raw)
        if talk and len(talk) > 200:
            with open(outpath, "w", encoding="utf-8") as f:
                f.write(talk)
            print(f"  OK {name} ({len(talk)} chars)")
        else:
            print(f"  WARN {name}: no content extracted")
    except Exception as e:
        print(f"  ERR {name}: {e}")
    time.sleep(1)

print(f"\nDone. Files in {TALKS_DIR}/:")
for f in sorted(os.listdir(TALKS_DIR)):
    size = os.path.getsize(os.path.join(TALKS_DIR, f))
    print(f"  {f} ({size:,} bytes)")
