import json
import asyncio
from typing import List

from main import recommend_assessments_raw

TRAIN_FILE = "train.json"   # labeled train set
K = 10


def normalize(urls: List[str]) -> List[str]:
    """Normalize URLs for fair comparison"""
    return [u.rstrip("/").lower() for u in urls]


def recall_at_k(predicted_urls: List[str], relevant_urls: List[str], k: int) -> float:
    predicted_top_k = predicted_urls[:k]
    if not relevant_urls:
        return 0.0
    hits = len(set(predicted_top_k) & set(relevant_urls))
    return hits / len(relevant_urls)


async def evaluate():
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    recalls = []

    for item in train_data:
        query = item["query"]

        # ✅ CORRECT KEY
        relevant_urls = normalize(item["relevant_urls"])

        # ✅ PURE VECTOR RETRIEVAL (NO LLM)
        try:
            recommendations = await recommend_assessments_raw(query)
        except RuntimeError as e:
            print(f"Runtime error while retrieving vectors: {e}")
            return 0.0

        predicted_urls = normalize([r["url"] for r in recommendations])

        r = recall_at_k(predicted_urls, relevant_urls, K)
        recalls.append(r)

        print(f"\nQuery: {query}")
        print(f"Recall@{K}: {r:.3f}")

    mean_recall = sum(recalls) / len(recalls)

    print("\n==============================")
    print(f"Mean Recall@{K}: {mean_recall:.4f}")
    print("==============================")

    return mean_recall


if __name__ == "__main__":
    asyncio.run(evaluate())
