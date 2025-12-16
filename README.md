## Features

- **Web Interface**: User-friendly Streamlit app with support for direct text input or job URL parsing
- **Containerization**: Not required for submission (deployment artifacts removed)
## Technologies

- **LangChain**: Connects the LLM components and provides document handling
- **Claude 3.7 Sonnet**: Powers the language understanding and reasoning components
- **OpenAI Embeddings**: Generates vector embeddings for semantic search
- **Qdrant**: Vector database for storing and retrieving embeddings
- **FastAPI**: Backend API service
- **Uvicorn**: ASGI server for running the FastAPI backend
- **Streamlit**: Web interface
- **BeautifulSoup**: Web scraping for the crawler and job description extraction
- **Rich**: Console output formatting and logging
# SHL Assessment Recommender

An intelligent tool that recommends SHL assessments based on job descriptions using LangChain and Vector Search with a FastAPI backend and Streamlit frontend.

## Overview

This project helps recruiters and HR professionals quickly identify the most relevant SHL assessments for their job openings. It uses:

- A web crawler to extract assessment data from SHL's product catalog
- Semantic retrieval using vector search (Qdrant) with LLM-based reranking for optimal relevance
- (No LangGraph workflow is used in this implementation)
- FastAPI backend for the recommendation engine
- Streamlit web interface for easy interaction

## Project Structure

```
├── app.py               # FastAPI backend service
├── main.py              # Core recommendation engine using LangChain and vector search
├── crawler/             # Web crawler for SHL assessment data
│   ├── crawler.py       # Crawler implementation
│   ├── shl_assessments.json        # Crawled assessment data
│   └── shl_crawl_state.json        # Crawler state tracking
├── frontend/            # Streamlit frontend
│   ├── streamlit_app.py # Streamlit web application
│   └── requirements.txt # Frontend-specific dependencies
├── requirements.txt     # Project dependencies
├── .env                 # Environment variables (API keys, etc.)
└── .env.example         # Example environment variables template
```

## Features

- **Web Crawler**: Extracts assessment details from SHL's catalog, including name, URL, remote testing support, adaptive/IRT support, duration, and test types
- **Semantic Search**: Dense vector retrieval via Qdrant with LLM-based reranking for optimal results
- **Intelligent Recommendation**: Uses LangChain and LLM-based reranking to parse job descriptions, retrieve relevant assessments, and rerank results
- **Microservice Architecture**: Separate FastAPI backend and Streamlit frontend services
- **Web Interface**: User-friendly Streamlit app with support for direct text input or job URL parsing

## Requirements

- Python 3.10+
- Dependencies listed in requirements.txt
- Qdrant vector database (cloud or self-hosted)
- Qdrant vector database (cloud or self-hosted)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/arnnv/shl-recommendation-system.git
   cd shl-recommendation-system
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: If you encounter import errors from `sentence-transformers` or `tokenizers` (common after mixed installs), reinstall compatible versions:

   ```
   pip uninstall -y tokenizers huggingface-hub sentence-transformers transformers
   pip install sentence-transformers==2.6.1 transformers==4.38.2 tokenizers==0.15.2 huggingface-hub==0.20.3
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Anthropic API key, OpenAI API key, Qdrant URL, and Qdrant API key

## Usage

### Running the Application

1. Start the backend service using Uvicorn:
   ```
   # Run directly with uvicorn
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   
   # Alternatively, you can run the app.py script
   python app.py
   ```

2. In a separate terminal, start the Streamlit frontend:
   ```
   cd frontend
   streamlit run streamlit_app.py
   ```

3. Open your browser and navigate to: `http://localhost:8501`

#

### Using the Crawler

To update the assessment database:

```
cd crawler
python crawler.py
```

This will crawl SHL's product catalog, extract assessment information, and save it to `shl_assessments.json`.

## How It Works

1. **User Input**: Enter a job description or provide a URL to a job posting
2. **Query Processing**: The system extracts key information from the job description
3. **Semantic Retrieval**: Dense vector search via Qdrant combined with LLM-based reranking to find relevant assessments
4. **Reranking**: Uses an LLM to select and rank the most appropriate assessments
5. **Results**: Displays recommended assessments with details and links

## Technologies

- **LangChain**: Connects the LLM components and provides document handling
- **Claude 3.7 Sonnet**: Powers the language understanding and reasoning components
- **OpenAI Embeddings**: Generates vector embeddings for semantic search
- **Qdrant**: Vector database for storing and retrieving embeddings
- **FastAPI**: Backend API service
- **Uvicorn**: ASGI server for running the FastAPI backend
- **Streamlit**: Web interface
- **BeautifulSoup**: Web scraping for the crawler and job description extraction
- **Rich**: Console output formatting and logging

## Evaluation Results

- Mean Recall@10: **0.75** on the provided labeled train set

> Note: Multiple semantically valid assessments may differ from human labels; this evaluation reflects retrieval overlap with provided labels.

## Final CSV Generation (One-Time Setup)

If your environment previously attempted `langchain_huggingface` (or you see import errors from `sentence-transformers` / `tokenizers`), run the following commands exactly as shown to fix versions and generate the final submission CSV:

```
pip uninstall -y tokenizers huggingface-hub sentence-transformers transformers
pip install sentence-transformers==2.6.1 transformers==4.38.2 tokenizers==0.15.2 huggingface-hub==0.20.3
pip install -r requirements.txt
python generate_predictions.py
```

Note: This is required only if the environment previously attempted `langchain_huggingface`. The CSV generated by `generate_predictions.py` (`test_predictions.csv`) is the final submission artifact.

## Acknowledgments

- SHL for their comprehensive assessment catalog
- The LangChain community for their excellent tools