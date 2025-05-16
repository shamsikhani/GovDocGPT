# GovDocGPT - Irish Statutory Instruments Q&A Application

## Overview

GovDocGPT is an AI-powered Question & Answering application focused on Irish Statutory Instruments (S.I.s). It leverages a Retrieval Augmented Generation (RAG) architecture to provide answers based on the content of these legal documents. Users can ask questions in natural language, and the system will retrieve relevant S.I. excerpts and generate a consolidated answer using OpenAI's language models.

## Features

*   **Retrieval Augmented Generation (RAG)**: Core architecture for providing context-aware answers.
*   **Web Crawling**: Fetches Irish Statutory Instruments from specified online sources (`crawler/main_crawler.py`).
*   **Document Processing**:
    *   Text extraction from HTML documents.
    *   Intelligent text chunking to prepare documents for embedding.
    *   Embedding generation using OpenAI's `text-embedding-3-small` model (`processing/document_processor.py`).
    *   Batch processing for efficient embedding generation, respecting API token limits.
*   **Semantic Search**: Utilizes cosine similarity (`numpy`) between user query embeddings and pre-computed document chunk embeddings to find the most relevant context.
*   **AI-Powered Answer Generation**: Employs OpenAI's chat completion models (e.g., GPT-3.5-turbo or similar) to synthesize answers based on retrieved S.I. chunks.
*   **FastAPI Backend**: A robust Python backend serves the Q&A API, handling user requests and orchestrating the RAG pipeline (`main_app.py`).
*   **Simple Web Frontend**: Basic HTML, CSS, and JavaScript interface for interacting with the application (`static/`).

## Technical Stack

*   **Backend**: Python
    *   **Framework**: FastAPI
    *   **ASGI Server**: Uvicorn
*   **AI & Machine Learning**:
    *   **OpenAI API**:
        *   `openai` Python library
        *   Embeddings Model: `text-embedding-3-small`
        *   Chat Completion Model: (e.g., `gpt-3.5-turbo`)
    *   **Numerical Operations**: `numpy` (for cosine similarity)
*   **Web Crawling**:
    *   `requests`
    *   `BeautifulSoup4`
*   **Environment Management**: `python-dotenv`
*   **Frontend**: HTML, CSS, JavaScript

## Project Structure

```
IrishEULaw/
├── .env                    # Stores environment variables (e.g., API keys) - NOT COMMITTED
├── .gitignore              # Specifies intentionally untracked files by Git
├── crawler/                # Scripts for web crawling S.I. documents
│   ├── __init__.py
│   └── main_crawler.py
├── data/                   # Data storage (mostly ignored by Git)
│   ├── crawled_docs/       # Raw HTML documents fetched by the crawler (gitignored)
│   └── processed_docs/     # Processed chunks with embeddings (gitignored)
├── main_app.py             # Main FastAPI application: API endpoints, RAG orchestration
├── processing/             # Scripts for document processing, chunking, and embedding generation
│   └── document_processor.py
├── qa_app.py               # Core Q&A logic: data loading, embedding comparison, OpenAI calls
├── requirements.txt        # Python project dependencies
├── static/                 # Frontend files
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── index.html
└── README.md               # This file
```

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/shamsikhani/GovDocGPT.git
    cd GovDocGPT
    ```

2.  **Create and Activate a Virtual Environment**:
    (Recommended, e.g., using `venv`)
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Create a `.env` file in the project root by copying `.env.example` (if it existed) or creating it manually:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Running the Application

The application requires a two-stage data preparation process before the server can be run.

1.  **Data Preparation - Step 1: Crawl Documents**:
    This script fetches S.I. documents and saves them to `data/crawled_docs/`.
    ```bash
    python crawler/main_crawler.py
    ```
    *(Ensure `crawler/main_crawler.py` is configured with the correct URLs and selectors if modified from its initial state.)*

2.  **Data Preparation - Step 2: Process and Embed Documents**:
    This script loads the crawled documents, chunks them, generates embeddings using OpenAI, and saves the result to `data/processed_docs/all_chunked_data_with_embeddings.json`.
    ```bash
    python processing/document_processor.py
    ```
    This step can take a significant amount of time and will make calls to the OpenAI API, incurring costs.

3.  **Start the FastAPI Server**:
    Once the data is prepared, start the backend server:
    ```bash
    python main_app.py
    ```
    By default, the server will run on `http://localhost:8010`.

4.  **Access the Application**:
    Open your web browser and navigate to `http://localhost:8010`.

## API Endpoint

*   **Endpoint**: `POST /api/ask`
*   **Request Body** (JSON):
    ```json
    {
        "question": "Your question about Irish S.I.s"
    }
    ```
*   **Response Body** (JSON):
    ```json
    {
        "answer": "AI-generated answer based on relevant S.I. chunks."
    }
    ```

## Key Design Decisions

*   **OpenAI Embeddings**: `text-embedding-3-small` was chosen for its balance of performance and cost for generating high-quality semantic embeddings.
*   **Batch Processing**: Implemented in `document_processor.py` to handle a large number of chunks efficiently when calling the OpenAI embeddings API, helping to avoid rate limits and manage token usage.
*   **Modular Components**: The application is structured with distinct modules for crawling, processing, Q&A logic, and serving, promoting maintainability.
*   **Direct API Interaction**: The application directly uses the `openai` Python library for fine-grained control over API calls for embeddings and chat completions.
*   **Stateless API**: The `/api/ask` endpoint is stateless, processing each request independently.

## Future Enhancements (Potential)

*   **Improved Error Handling**: More granular error handling and user feedback on the frontend.
*   **Streaming Responses**: Implement streaming for the `/api/ask` endpoint to provide faster initial feedback to the user for long-generated answers.
*   **Advanced Frontend**: Enhance the UI/UX with features like conversation history, document source display, and feedback mechanisms.
*   **Optimized Data Storage/Retrieval**: For very large datasets, explore vector databases (e.g., Pinecone, Weaviate, FAISS) instead of loading all embeddings into memory from a JSON file.
*   **Caching**: Implement caching for OpenAI API calls or frequently asked questions to reduce latency and cost.
*   **Evaluation Framework**: Develop a framework for evaluating the quality of answers and retrieval accuracy.
*   **User Authentication**: If exposing publicly or for multiple users, add authentication.
*   **Admin Interface**: For managing documents, monitoring, or re-triggering processing.