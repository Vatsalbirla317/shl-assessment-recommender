import pandas as pd
import json

# Load Excel
xls = pd.ExcelFile("Gen_AI Dataset.xlsx")

# ---------- TRAIN ----------
train_df = pd.read_excel(xls, "Train-Set")

train_data = []
for query, group in train_df.groupby("Query"):
    train_data.append({
        "query": query,
        "relevant_urls": group["Assessment_url"].dropna().tolist()
    })

with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)

print(f"Saved train.json with {len(train_data)} queries")

# ---------- TEST ----------
test_df = pd.read_excel(xls, "Test-Set")

test_data = [{"query": q} for q in test_df["Query"].dropna().tolist()]

with open("test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2)

print(f"Saved test.json with {len(test_data)} queries")
