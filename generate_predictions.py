# generate_predictions.py
import csv
import json
import asyncio
from main import recommend_assessments

TEST_FILE = "test.json"              # unlabeled test set
OUTPUT_FILE = "test_predictions.csv"


async def generate():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    rows = []

    for item in test_queries:
        query = item["query"]

        try:
            recommendations = await recommend_assessments(query)
        except RuntimeError as e:
            raise RuntimeError(f"Runtime error while generating recommendations: {e}")

        # Extract URLs, remove empties, preserve order, and deduplicate per query
        seen = set()
        unique_urls = []
        for rec in recommendations:
            url = rec.get("url")
            if not url or not str(url).strip():
                continue
            u = str(url).strip()
            if u not in seen:
                seen.add(u)
                unique_urls.append(u)

        # Enforce 5-10 URLs per query
        if len(unique_urls) < 5:
            raise AssertionError(f"Less than 5 unique recommendations for query: '{query}' (found {len(unique_urls)})")
        if len(unique_urls) > 10:
            # Trim to 10 but raise loudly so user knows
            raise AssertionError(f"More than 10 recommendations for query: '{query}' (found {len(unique_urls)})")

        # Add rows for this query (one row per URL)
        for url in unique_urls:
            rows.append({"Query": query, "Assessment_url": url})

    # Final check: no empty rows and header correctness
    if not rows:
        raise AssertionError("No prediction rows to write")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Query", "Assessment_url"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(generate())
