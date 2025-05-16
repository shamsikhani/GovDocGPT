import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import re

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
DATA_FILE_PATH = os.path.join("data", "processed_docs", "all_chunked_data_with_embeddings.json")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL_NAME = 'gpt-4-turbo-preview'
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.2
SIMILARITY_THRESHOLD_SPECIFIC = 0.2
TOP_K_FALLBACK_CHUNKS = 3

# --- FastAPI App Setup ---
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Global Variables ---
si_chunks_data = []
openai_client = None

class AskRequest(BaseModel):
    question: str

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def load_data_and_model():
    global si_chunks_data, openai_client
    print("--- load_data_and_model: START --- ", flush=True)
    print(f"Loading processed S.I. data from {DATA_FILE_PATH}...", flush=True)
    try:
        print("--- load_data_and_model: Before json.load() --- ", flush=True)
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_loaded_data = json.load(f)
        print("--- load_data_and_model: After json.load() --- ", flush=True)
        si_chunks_data = raw_loaded_data
        print(f"--- load_data_and_model: Starting loop to check {len(si_chunks_data)} chunks... ---", flush=True)
        for i, chunk in enumerate(si_chunks_data):
            if 'embedding' not in chunk or not isinstance(chunk['embedding'], list):
                print(f"Warning: Chunk {i} (ID: {chunk.get('id', 'N/A')}) missing or has invalid embedding format.", flush=True)
        print(f"--- load_data_and_model: Finished loop. Loaded {len(si_chunks_data)} S.I. chunks. ---", flush=True)
    except FileNotFoundError:
        print(f"FATAL ERROR: Processed data file not found at {DATA_FILE_PATH}.", flush=True)
        raise RuntimeError(f"Processed data file not found: {DATA_FILE_PATH}")
    except json.JSONDecodeError:
        print(f"FATAL ERROR: Could not decode JSON from {DATA_FILE_PATH}.", flush=True)
        raise RuntimeError(f"Could not decode JSON: {DATA_FILE_PATH}")

    print("--- load_data_and_model: Before os.getenv(OPENAI_API_KEY) --- ", flush=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("FATAL ERROR: OPENAI_API_KEY not found in environment variables. OpenAI calls will fail.", flush=True)
        raise RuntimeError("OPENAI_API_KEY not found.")
    else:
        print("--- load_data_and_model: Before OpenAI client init --- ", flush=True)
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client configured with API key.", flush=True)
    print("--- load_data_and_model: END --- ", flush=True)

@app.on_event("startup")
def startup_event():
    load_data_and_model()

def find_relevant_chunks(user_question: str):
    print("Calling find_relevant_chunks...")
    if not si_chunks_data or openai_client is None:
        print("Data or OpenAI client not loaded yet.")
        raise HTTPException(status_code=503, detail="Server not fully initialized. Data or AI model unavailable.")

    try:
        response = openai_client.embeddings.create(
            input=[user_question],
            model=OPENAI_EMBEDDING_MODEL
        )
        query_embedding_list = response.data[0].embedding
        query_embedding = np.array(query_embedding_list, dtype=np.float32)
    except Exception as e:
        print(f"Error getting query embedding from OpenAI: {e}")
        raise HTTPException(status_code=503, detail=f"Error generating query embedding: {str(e)}")

    si_match = re.search(r"(?:S\.I\.|SI)(?:\s*No\.?)?\s*(\d+)\s*(?:of|/|\s)\s*(\d{4})", user_question, re.IGNORECASE)
    
    specific_si_number_query = None
    relevant_s_i_items = []

    if si_match:
        specific_si_number_query = f"{si_match.group(1)}/{si_match.group(2)}"
        print(f"Specific S.I. detected: {specific_si_number_query}")
        
        specific_si_chunks_with_indices = [
            (i, chunk) for i, chunk in enumerate(si_chunks_data)
            if chunk.get('si_number') == specific_si_number_query and 'embedding' in chunk
        ]

        if specific_si_chunks_with_indices:
            scores = []
            for original_idx, chunk_data in specific_si_chunks_with_indices:
                doc_embedding = np.array(chunk_data['embedding'], dtype=np.float32)
                sim = cosine_similarity(query_embedding, doc_embedding)
                scores.append({'original_index': original_idx, 'score': sim, 'chunk_data': chunk_data})
            
            sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
            
            for item in sorted_scores:
                if item['score'] >= SIMILARITY_THRESHOLD_SPECIFIC and len(relevant_s_i_items) < TOP_K_RESULTS:
                    chunk_info = item['chunk_data']
                    relevant_s_i_items.append({
                        'si_number': chunk_info.get('si_number'),
                        'year': chunk_info.get('year'),
                        'title': chunk_info.get('title'),
                        'text': chunk_info.get('text'),
                        'url': chunk_info.get('url'),
                        'score': item['score']
                    })
        else:
            print(f"No chunks found for specific S.I. {specific_si_number_query} or they lack embeddings.")

    if not relevant_s_i_items or len(relevant_s_i_items) < TOP_K_FALLBACK_CHUNKS:
        print("Performing general semantic search across all S.I. chunks...")
        already_added_texts = {item['text'] for item in relevant_s_i_items}
        
        scores = []
        for i, chunk_data in enumerate(si_chunks_data):
            if 'embedding' not in chunk_data:
                continue
            if chunk_data.get('text') in already_added_texts:
                continue
            
            doc_embedding = np.array(chunk_data['embedding'], dtype=np.float32)
            sim = cosine_similarity(query_embedding, doc_embedding)
            scores.append({'original_index': i, 'score': sim, 'chunk_data': chunk_data})

        sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
        
        for item in sorted_scores:
            if len(relevant_s_i_items) >= TOP_K_RESULTS:
                break
            if item['score'] >= SIMILARITY_THRESHOLD and item['chunk_data'].get('text') not in already_added_texts:
                chunk_info = item['chunk_data']
                relevant_s_i_items.append({
                    'si_number': chunk_info.get('si_number'),
                    'year': chunk_info.get('year'),
                    'title': chunk_info.get('title'),
                    'text': chunk_info.get('text'),
                    'url': chunk_info.get('url'),
                    'score': item['score']
                })
                already_added_texts.add(chunk_info.get('text'))

    relevant_s_i_items = sorted(relevant_s_i_items, key=lambda x: x['score'], reverse=True)
    
    print(f"Total relevant chunks found: {len(relevant_s_i_items)}")
    return relevant_s_i_items

def get_openai_response_sync(user_question: str, relevant_si_texts: list):
    if not openai_client:
        print("OpenAI client not initialized in get_openai_response_sync")
        raise HTTPException(status_code=503, detail="AI model client not available.")

    if not relevant_si_texts:
        print("No relevant S.I. texts provided to OpenAI, returning predefined message.")
        return "I couldn't find any S.I.s that seem directly relevant to your question based on the current data. Please try rephrasing or asking about a different topic."

    context_str = "\n\n---\n\n".join([
        (
            f"Excerpt from S.I. {item['si_number']}/{item['year']} - {item['title']}:\n{item['text']}\n(Source URL: {item['url'] if item.get('url') else 'N/A'}) (Relevance score: "
            + (f"{item['score']:.4f}" if isinstance(item.get('score'), (float, int)) else str(item.get('score', 'N/A')))
            + ")"
        )
        for item in relevant_si_texts
    ])
    
    system_prompt = (
        "You are a helpful AI assistant specializing in Irish Statutory Instruments (S.I.s). "
        "You will be given a user's question and excerpts from relevant S.I. documents. "
        "Answer the user's question in a conversational tone, based *only* on the provided S.I. excerpts. "
        "Clearly cite the S.I. number and year (e.g., S.I. No. 123/2023) for any information you use. "
        "If the provided excerpts do not contain enough information to answer the question, say so explicitly. "
        "Do not make up information or answer based on external knowledge."
    )
    user_message_content = f"User Question: {user_question}\n\nRelevant S.I. Excerpts:\n{context_str}"

    print("Attempting to call OpenAI Chat API...")
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message_content}
            ],
            temperature=0.2,
        )
        assistant_response = response.choices[0].message.content
        print("--- Successfully received response from OpenAI Chat API ---")
        return assistant_response.strip()
    except Exception as e:
        print(f"!!! EXCEPTION in get_openai_response_sync during OpenAI call: {type(e).__name__}: {e} !!!")
        raise HTTPException(status_code=503, detail=f"Error communicating with AI model: {str(e)}")

@app.post("/api/ask")
def api_ask(request_data: AskRequest):
    print("--- Endpoint /api/ask called ---", flush=True)
    user_question = request_data.question
    print(f"Received question in /api/ask: {user_question}", flush=True)
    if not user_question:
        print("Question is empty, raising HTTPException 400.", flush=True)
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        print("Calling find_relevant_chunks...", flush=True)
        relevant_chunks = find_relevant_chunks(user_question)
        
        if not relevant_chunks:
            print("No relevant S.I. chunks found by find_relevant_chunks.", flush=True)
        else:
            print(f"Found {len(relevant_chunks)} relevant chunks.", flush=True)

        print("Calling get_openai_response_sync...", flush=True)
        ai_response = get_openai_response_sync(user_question, relevant_chunks)
        print("--- Endpoint /api/ask successfully returning response ---", flush=True)
        return {"answer": ai_response}
    except HTTPException as http_exc:
        print(f"!!! HTTPException in /api/ask: {http_exc.status_code} - {http_exc.detail} !!!", flush=True)
        raise http_exc
    except Exception as e:
        print(f"!!! UNEXPECTED EXCEPTION in /api/ask: {type(e).__name__}: {e} !!!", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/health")
def health_check():
    print("--- Endpoint /health called ---", flush=True)
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    index_html_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    try:
        with open(index_html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

if __name__ == "__main__":
    print("Starting Uvicorn server...")
    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL: OPENAI_API_KEY is not set. The application will not work.")
        print("Please set the OPENAI_API_KEY environment variable.")
    uvicorn.run(app, host="0.0.0.0", port=8010) # Changed port to 8010
