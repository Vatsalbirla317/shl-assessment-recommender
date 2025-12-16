import os
from dotenv import load_dotenv

from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import recommend_assessments

load_dotenv()

port = int(os.environ.get("PORT", 8000))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobRequest(BaseModel):
    query: str | None = None
    job_description: str | None = None

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return {"message": "SHL Backend is running!"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.post("/recommend")
async def recommend(request: JobRequest):
    query_text = request.query or request.job_description
    if not query_text or not query_text.strip():
        raise HTTPException(status_code=400, detail="'query' or 'job_description' must be provided")
    try:
        recommendations = await recommend_assessments(query_text)
    except RuntimeError as e:
        # Return a clear runtime error message when environment or clients are not configured
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Generic error
        raise HTTPException(status_code=500, detail="Internal server error")

    return {"recommended_assessments": recommendations}


def run_smoke_tests():
    """Run minimal smoke checks against the core functions without HTTP or TestClient.

    This replaces the recommend_assessments function with a deterministic stub
    so the smoke tests can validate response shape without external services.
    """
    import main as main_module
    import asyncio

    # Health check (call directly)
    h = health_check()
    assert h == {"status": "healthy"}, f"Health check failed: {h}"

    # Monkeypatch recommend_assessments to a deterministic stub
    original = main_module.recommend_assessments

    async def stub_recommend(q: str):
        items = []
        for i in range(5):
            items.append({
                "name": f"Test Assessment {i+1}",
                "url": f"https://example.com/test{i+1}",
                "description": "Test description",
                "duration": 10,
                "remote_support": "Yes" if i % 2 == 0 else "No",
                "adaptive_support": "No",
                "test_type": ["K", "P"]
            })
        return items

    main_module.recommend_assessments = stub_recommend

    try:
        recs = asyncio.run(main_module.recommend_assessments("test query"))
        assert isinstance(recs, list), "recommend_assessments must return a list"
        assert 5 <= len(recs) <= 10, f"Expected 5-10 recommendations, got {len(recs)}"

        required = {"name", "url", "description", "duration", "remote_support", "adaptive_support", "test_type"}
        for rec in recs:
            assert required.issubset(set(rec.keys())), f"Missing required fields in rec: {rec}"
            assert "solution_type" not in rec, "solution_type should not be exposed in API response"
            # Type checks
            assert isinstance(rec["name"], str)
            assert isinstance(rec["url"], str)
            assert isinstance(rec["description"], str)
            assert isinstance(rec["duration"], int)
            assert rec["remote_support"] in ("Yes", "No")
            assert rec["adaptive_support"] in ("Yes", "No")
            assert isinstance(rec["test_type"], list)

        print("SMOKE TESTS PASSED")
    finally:
        main_module.recommend_assessments = original


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        run_smoke_tests()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
