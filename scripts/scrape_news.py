#!/usr/bin/env python3
"""Collect unseen Fox News and NBC article URLs using public RSS feeds and article sitemaps.

Python 3.9+, no third-party packages needed.

From the repository root, exclude the supplied starter CSV explicitly:
    python scripts/scrape_news.py --exclude course/url_train.csv --output data/url_new.csv

Paths may also be supplied for another CSV and output location. Without
--exclude, the script looks for url_train.csv in the current working directory.

Run again periodically to append newly discovered URLs. 'New' means not
already in the exclusion CSV(s) or output CSV, not a publication-date cutoff.
Defaults to 5,000 new URLs per run (2,500 per source), including older articles.
Use --target 10000 for a larger collection. Stops if sitemaps are exhausted
or --max-sitemaps (default 100 per source) is reached. Run one instance at a time.

The output matches the training file: one 'url' column of FULL links. Keep
the host to infer labels. In your own training pipeline, keep only the final
headline segment and remove words containing digits before model training.
Never train on the raw full links: their domains reveal the answer.
"""

import argparse
import csv
import gzip
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, deque
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qs


FEEDS = {
    "FoxNews": "https://moxie.foxnews.com/google-publisher/latest.xml",
    "NBC": "https://feeds.nbcnews.com/nbcnews/public/news",
}
SITEMAPS = {
    "FoxNews": "https://www.foxnews.com/sitemap.xml",
    "NBC": "https://www.nbcnews.com/sitemap/nbcnews/sitemap-index",
}
HOSTS = {"foxnews.com": "FoxNews", "nbcnews.com": "NBC"}


def normalize_url(url):
    """Remove tracking queries/fragments and normalize host/scheme for dedup."""
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower().removeprefix("www.")
        if parts.scheme not in ("http", "https") or host not in HOSTS:
            return None
        path = parts.path.rstrip("/")
        if not path:
            return None
        return urlunsplit(("https", "www." + host, path, "", ""))
    except ValueError:
        return None


def read_urls(path):
    """Validate the schema before appending anything to an existing file."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["url"]:
            raise ValueError(f"{path}: expected exactly one column named 'url'")
        return {clean for row in reader if (clean := normalize_url(row["url"] or ""))}


def fetch_xml(feed_url):
    """Fetch RSS or sitemap XML with bounded retries."""
    request = urllib.request.Request(
        feed_url,
        headers={"User-Agent": "NewsSourceClassProject/1.0 (RSS reader)",
                 "Accept": "application/rss+xml, application/xml, text/xml"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            if data.startswith(b"\x1f\x8b"):
                data = gzip.decompress(data)
            return ET.fromstring(data)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** (attempt + 1))



def article_url(url):
    """Exclude section/index/video URLs; retain headline-like article paths."""
    parts = urlsplit(url)
    segments = parts.path.strip("/").split("/")
    if len(segments) < 2 or any(x in segments for x in ("video", "slideshow")):
        return False
    tail = segments[-1]
    if parts.hostname.endswith("nbcnews.com"):
        return bool(re.search(r"-(?:rcna|ncna|n)\d+$", tail))
    return len(tail.split("-")) >= 4 and not tail.endswith(".xml")


def candidates(source, max_sitemaps, delay):
    """Yield recent RSS links, then traverse sitemap indexes newest first."""
    try:
        root = fetch_xml(FEEDS[source])
        for item in root.findall("./channel/item"):
            yield item.findtext("link", default="")
    except (OSError, ValueError, ET.ParseError) as error:
        print(f"WARNING: {source} RSS: {error}", file=sys.stderr)
    pending = deque([SITEMAPS[source]])
    visited = set()
    while pending and len(visited) < max_sitemaps:
        sitemap = pending.popleft()
        if sitemap in visited:
            continue
        visited.add(sitemap)
        time.sleep(delay)
        try:
            root = fetch_xml(sitemap)
            kind = root.tag.rsplit("}", 1)[-1]
            if kind not in ("sitemapindex", "urlset"):
                raise ValueError("Not a sitemap XML document")
            entries = []
            for child in root:
                fields = {e.tag.rsplit("}", 1)[-1]: e.text or "" for e in child}
                if fields.get("loc"):
                    entries.append((fields.get("lastmod", ""), fields["loc"]))
            entries.sort(reverse=True)
            print(f"{source}: sitemap {len(visited)}/{max_sitemaps}, "
                  f"{len(entries)} entries", flush=True)
            if kind == "sitemapindex":
                # Depth-first traversal avoids fetching every index before articles.
                children = [loc for _, loc in entries
                            if urlsplit(loc).scheme in ("http", "https")
                            and urlsplit(loc).hostname == urlsplit(sitemap).hostname]
                if source == "FoxNews":
                    children = [loc for loc in children
                                if parse_qs(urlsplit(loc).query).get("type") == ["articles"]]
                    children.sort(key=lambda loc: int(parse_qs(urlsplit(loc).query).get("page", ["1"])[0]))
                else:
                    children = [loc for loc in children if loc.endswith("-article.xml")]
                    children.sort(reverse=True)
                pending.extendleft(reversed(children))
            else:
                for _, loc in entries:
                    yield loc
        except (OSError, ValueError, ET.ParseError) as error:
            print(f"WARNING: {sitemap}: {error}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("url_new.csv"))
    parser.add_argument("--exclude", type=Path, nargs="+",
                        help="Existing single-column URL CSVs to skip; defaults to url_train.csv if present")
    parser.add_argument("--target", type=int, default=5000,
                        help="New URLs to add this run, split equally by source (default: 5000)")
    parser.add_argument("--max-sitemaps", type=int, default=100,
                        help="Maximum sitemap requests per source (default: 100)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between sitemap requests (default: 0.5)")
    args = parser.parse_args()
    if args.target < 1 or args.max_sitemaps < 1 or args.delay < 0:
        parser.error("target and max-sitemaps must be positive; delay must be nonnegative")
    exclusions = args.exclude
    if exclusions is None:
        exclusions = [Path("url_train.csv")] if Path("url_train.csv").exists() else []
        if not exclusions:
            print("Note: no url_train.csv found; deduplicating against output only.", file=sys.stderr)
    seen = set()
    try:
        for path in exclusions:
            seen.update(read_urls(path))
        if args.output.exists() and args.output.stat().st_size:
            seen.update(read_urls(args.output))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    new_urls = []
    counts = Counter()
    for index, source in enumerate(FEEDS):
        quota = args.target // 2 + (args.target % 2 if index == 0 else 0)
        if not quota:
            continue
        for link in candidates(source, args.max_sitemaps, args.delay):
            url = normalize_url(link)
            if not url or url in seen or not article_url(url):
                continue
            host = urlsplit(url).hostname.removeprefix("www.")
            if HOSTS[host] != source:
                continue
            seen.add(url)
            new_urls.append(url)
            counts[source] += 1
            if counts[source] >= quota:
                break
        print(f"{source}: collected {counts[source]}/{quota} new URLs", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not args.output.exists() or args.output.stat().st_size == 0
    # Preserve a valid existing CSV even if its final row has no newline.
    needs_newline = False
    if not needs_header:
        with args.output.open("rb") as handle:
            handle.seek(-1, 2)
            needs_newline = handle.read(1) not in (b"\n", b"\r")
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(["url"])
        writer.writerows([url] for url in new_urls)
    print(f"Added {len(new_urls)} URLs to {args.output}: "
          f"FoxNews={counts['FoxNews']}, NBC={counts['NBC']}")
    if len(new_urls) < args.target:
        print("Target not reached: available unique URLs, request failures, or the sitemap "
              "limit constrained collection. Try increasing --max-sitemaps.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
