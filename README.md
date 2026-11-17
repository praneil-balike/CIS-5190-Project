# CIS 5190 News Source Classification

This repository holds the course starter materials for the Fall 2026 CIS 4190/5190 Applied Machine Learning Project B, plus a scraper for collecting additional article URLs. The eventual task is to classify a headline slug as originating from **Fox News** or **NBC News**.

The hidden evaluation input contains only the headline-like final URL segment. The domain, news section, and words containing digits are removed. Training and validation must use the same information; full URLs are useful for deriving labels and collecting data, but their domains must not be model features.

## Repository layout

| Path | Contents |
| --- | --- |
| `course/url_train.csv` | Supplied starter URLs. Keep this source file intact. |
| `course/templates/` | Unimplemented `model.py` and `preprocess.py` examples from the course. |
| `course/baseline/` | Supplied tested baseline code and its `model.pt` weights. |
| `course/eval_project_b.py` | Supplied local evaluation script. |
| `scripts/scrape_news.py` | Existing RSS and sitemap scraper for additional Fox News and NBC News article URLs. |

The course templates and baseline are reference material; this repository does not yet contain a project model or training pipeline.

## Collect additional URLs

The scraper uses Python 3.9+ and the standard library. From the repository root, run:

```bash
python scripts/scrape_news.py --exclude course/url_train.csv --output data/url_new.csv
```

This writes a single-column `url` CSV with full article links, excluding URLs already in the starter CSV or existing output. Repeated runs append newly found URLs. The script collects from public RSS feeds and article sitemaps; see its module docstring for limits and options. The output directory is created automatically. No additional URLs have been collected as part of this repository setup.

## Later submission contract

The leaderboard submission consists of `preprocess.py`, `model.py`, and, if trained weights are used, a compatible PyTorch state dict in `model.pt` or `model.pth`. `prepare_data(csv_path)` must return one input per CSV row in the original order, including hidden rows with no label. Inference runs on CPU under the course runtime limit. The included evaluator can check the final implementation against headline-only inputs.

The course handout and provided evaluator are authoritative for final interface and submission details.
