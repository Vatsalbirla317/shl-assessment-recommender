import asyncio
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
# SentenceTransformerEmbeddings will be imported lazily in get_embedding_model() to avoid import-time dependency failures
from langchain_openai import ChatOpenAI

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()

# Environment variables will be read lazily when clients are initialized
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SHL_FILE = "shl_assessments.json"
COLLECTION_NAME = "shl_assessments"

# --------------------------------------------------
# LOCAL EMBEDDINGS (FREE, STABLE)
# --------------------------------------------------
# Lazy initialization of SentenceTransformerEmbeddings to avoid import-time dependency issues
_embedding_model = None

def get_embedding_model():
    """Return a cached SentenceTransformerEmbeddings instance (model: all-MiniLM-L6-v2)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from langchain_community.embeddings import SentenceTransformerEmbeddings as _STE
            _embedding_model = _STE(model_name="all-MiniLM-L6-v2")
        except ImportError as e:
            # Provide a clear actionable error message for common version mismatch issues
            raise RuntimeError(
                "Failed to import sentence-transformers or compatible tokenizer/transformers versions. "
                "This is commonly caused by incompatible versions of tokenizers/transformers installed. "
                "To fix, run:\n\n"
                "pip uninstall -y tokenizers huggingface-hub sentence-transformers transformers\n"
                "pip install sentence-transformers==2.6.1 transformers==4.38.2 tokenizers==0.15.2 huggingface-hub==0.20.3\n\n"
                "After reinstalling, retry running the script."
                f"\nOriginal import error: {e}"
            ) from e
    return _embedding_model

# --------------------------------------------------
# LOAD SHL DATA
# --------------------------------------------------
with open(SHL_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Ensure 'solution_type' is present for all entries (default to 'individual') and persist back
_updated = False
for entry in data:
    if 'solution_type' not in entry:
        entry['solution_type'] = 'individual'
        _updated = True

if _updated:
    with open(SHL_FILE, 'w', encoding='utf-8') as wf:
        json.dump(data, wf, indent=2, ensure_ascii=False)

documents = []
for entry in data:
    content = f"""
    Name: {entry.get('name', '')}
    Description: {entry.get('description', '')}
    Test Types: {', '.join(entry.get('test_types', []))}
    Duration: {entry.get('duration', '')}
    Remote Support: {entry.get('remote_testing_support', '')}
    Adaptive Support: {entry.get('adaptive_irt_support', '')}
    """
    documents.append(Document(page_content=content, metadata=entry))

# --------------------------------------------------
# LAZY QDRANT INITIALIZATION  ✅ FIX
# --------------------------------------------------
_retriever = None

def get_retriever():
    global _retriever

    if _retriever is not None:
        return _retriever

    # Initialize Qdrant client lazily and fail at runtime if not configured
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set to use the vector store")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

    try:
        client.get_collection(COLLECTION_NAME)
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=get_embedding_model(),
        )
    except UnexpectedResponse:
        vectorstore = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=get_embedding_model(),
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
        )

    # Retrieve a larger candidate pool (k=20) so downstream reranker can pick 5-10
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    return _retriever

# --------------------------------------------------
# RAW RETRIEVAL (NO LLM) — FOR EVALUATION
# --------------------------------------------------
async def recommend_assessments_raw(job_description: str, k: int = 10):
    """
    Pure vector retrieval.
    Used ONLY for Recall@10 evaluation.
    """
    retriever = get_retriever()
    docs = await retriever.ainvoke(job_description)
    # Filter to individual solutions only (treat missing field as individual)
    filtered = [d for d in docs if d.metadata.get('solution_type', 'individual') == 'individual']
    if not filtered:
        filtered = docs

    return [
        {"url": doc.metadata.get("url", "").rstrip("/").lower()}
        for doc in filtered[:k]
    ]

# --------------------------------------------------
# GROQ LLM (RERANKING ONLY)
# --------------------------------------------------
_llm = None

def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY must be set to use the reranker LLM")

    _llm = ChatOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    return _llm

# --------------------------------------------------
# RERANK PROMPT
# --------------------------------------------------
rerank_prompt = PromptTemplate.from_template("""
You are selecting the most relevant SHL assessments.

Job requirement:
{query}

Assessments:
{assessments}

Return ONLY a JSON array of indices (1-based).
Maximum 10.

Example:
[1, 3, 5]
""")

# --------------------------------------------------
# MAIN RECOMMENDATION FUNCTION (API)
# --------------------------------------------------
async def recommend_assessments(job_description: str) -> List[Dict[str, Any]]:
    retriever = get_retriever()
    # Get the initial candidate set (up to 20) from vector search
    retrieved_docs = await retriever.ainvoke(job_description)

    # Filter only individual solutions (default to 'individual' when missing)
    filtered_docs = [d for d in retrieved_docs if d.metadata.get('solution_type', 'individual') == 'individual']
    if not filtered_docs:
        # If no docs survived filtering, fall back to original retrieved_docs
        filtered_docs = retrieved_docs

    blocks = []
    for i, doc in enumerate(filtered_docs):
        blocks.append(
            f"""
            Assessment {i+1}:
            Name: {doc.metadata.get('name')}
            Description: {doc.metadata.get('description')}
            Test Types: {doc.metadata.get('test_types')}
            Duration: {doc.metadata.get('duration')}
            """
        )

    prompt = rerank_prompt.format(
        query=job_description,
        assessments="\n".join(blocks),
    )

    llm = get_llm()
    response = (await llm.ainvoke(prompt)).content.strip()

    try:
        indices = json.loads(response)
        if not isinstance(indices, list):
            raise ValueError()
    except Exception:
        # Default to top 5 if parser fails
        indices = list(range(1, min(6, len(filtered_docs) + 1)))

    # Normalize indices: unique, valid, and at most 10
    seen = set()
    clean_indices = []
    for idx in indices:
        try:
            i = int(idx)
        except Exception:
            continue
        if 1 <= i <= len(filtered_docs) and i not in seen:
            clean_indices.append(i)
            seen.add(i)
        if len(clean_indices) >= 10:
            break

    # If reranker returned fewer than 5, fill from filtered_docs by vector order
    if len(clean_indices) < 5:
        for i in range(1, len(filtered_docs) + 1):
            if i not in seen:
                clean_indices.append(i)
                seen.add(i)
            if len(clean_indices) >= 5:
                break

    # Truncate to a maximum of 10 recommendations
    clean_indices = clean_indices[:10]

    recommendations = []
    for idx in clean_indices:
        if 1 <= idx <= len(filtered_docs):
            doc = filtered_docs[idx - 1]
            recommendations.append({
                # API spec compliance: exact fields and types
                "name": doc.metadata.get("name", ""),
                "url": doc.metadata.get("url", ""),
                "description": doc.metadata.get("description", ""),
                # Parse duration into integer minutes
                "duration": parse_duration_minutes(doc.metadata.get("duration", None)),
                "remote_support": normalize_yes_no(doc.metadata.get("remote_testing_support", doc.metadata.get("remote_support", "No"))),
                "adaptive_support": normalize_yes_no(doc.metadata.get("adaptive_irt_support", doc.metadata.get("adaptive_support", "No"))),
                # Convert verbose test type names to letter codes
                "test_type": map_test_types_to_codes(doc.metadata.get("test_types", [])),
            })

    return recommendations


def parse_duration_minutes(duration_field: str) -> int:
    """Parse duration string like '49 minutes' into integer minutes. Return 0 if not parsable."""
    if not duration_field:
        return 0
    if isinstance(duration_field, int):
        return duration_field
    try:
        import re
        m = re.search(r"(\d+)", str(duration_field))
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 0


def normalize_yes_no(val: str) -> str:
    if not val:
        return "No"
    v = str(val).strip().lower()
    if v in ("yes", "y", "true", "1"):
        return "Yes"
    return "No"


def map_test_types_to_codes(types) -> List[str]:
    """Map verbose test types to letter codes expected by the spec."""
    if not types:
        return []
    reverse_map = {
        "Ability": "A",
        "Behavioral": "B",
        "Cognitive": "C",
        "Knowledge": "K",
        "Personality": "P",
        "Situational": "S",
        # Accept already-single-letter entries
        "A": "A",
        "B": "B",
        "C": "C",
        "K": "K",
        "P": "P",
        "S": "S",
    }
    codes = []
    for t in types:
        t_str = str(t).strip()
        code = reverse_map.get(t_str)
        if not code:
            # Try capitalized first letter fallback
            t_up = t_str.title()
            code = reverse_map.get(t_up)
        if code and code not in codes:
            codes.append(code)
    return codes

# --------------------------------------------------
# LOCAL TEST
# --------------------------------------------------
if __name__ == "__main__":
    jd = "Hiring a Java developer who collaborates with stakeholders"
    out = asyncio.run(recommend_assessments(jd))
    print(json.dumps(out, indent=2))
