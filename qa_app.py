import os
import json
import numpy as np
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'processed_docs', 'all_chunked_data_with_embeddings.json')
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_MODEL_NAME = 'gpt-3.5-turbo'
TOP_K_RESULTS = 3

# --- Global Variables ---
si_chunks_data = []
openai_client = None

def load_data_and_model():
    global si_chunks_data, openai_client

    print(f"Loading processed S.I. data from {PROCESSED_DATA_PATH}...")
    try:
        with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
            si_chunks_data = json.load(f)
        print(f"Loaded {len(si_chunks_data)} S.I. chunks.")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}. Please run document_processor.py first.")
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {PROCESSED_DATA_PATH}.")
        return False

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the application. For example:")
        print("  PowerShell: $env:OPENAI_API_KEY = \"your_key\"")
        print("  Bash/Zsh:   export OPENAI_API_KEY=\"your_key\"")
        return False
    try:
        openai_client = openai.OpenAI(api_key=api_key)
        print("OpenAI client initialized.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return False

    return True


def find_relevant_chunks(user_question):
    if not si_chunks_data or not openai_client:
        print("Data or OpenAI client not loaded. Cannot find relevant chunks.")
        return []

    try:
        response = openai_client.embeddings.create(input=[user_question], model=OPENAI_EMBEDDING_MODEL)
        question_embedding = np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error generating query embedding using OpenAI: {e}")
        return []
    
    chunk_embeddings_list = [chunk['embedding'] for chunk in si_chunks_data if 'embedding' in chunk]
    if not chunk_embeddings_list:
        print("No embeddings found in the loaded S.I. data.")
        return []
    
    chunk_embeddings_np = np.array(chunk_embeddings_list)

    dot_products = np.dot(chunk_embeddings_np, question_embedding)
    norm_question = np.linalg.norm(question_embedding)
    norm_chunks = np.linalg.norm(chunk_embeddings_np, axis=1)
    
    cosine_similarities = np.zeros_like(dot_products, dtype=float)
    
    valid_indices = (norm_question > 1e-9) & (norm_chunks > 1e-9) 
    
    cosine_similarities[valid_indices] = dot_products[valid_indices] / (norm_question * norm_chunks[valid_indices])
        
    top_k_indices = np.argsort(-cosine_similarities)[:TOP_K_RESULTS]
    
    relevant_chunks = []
    for idx in top_k_indices:
        chunk = si_chunks_data[idx]
        relevant_chunks.append({
            'text': chunk.get('text'),
            'si_number': chunk.get('si_number'),
            'title': chunk.get('title'),
            'year': chunk.get('year'),
            'url': chunk.get('url'),
            'score': cosine_similarities[idx].item() 
        })
    return relevant_chunks


def get_openai_response(user_question, relevant_si_texts):
    if not openai_client:
        print("OpenAI client not initialized.")
        return "Sorry, the AI model is not available at the moment."
        
    if not relevant_si_texts:
        return "I couldn't find any specific information in the statutory instruments related to your question. Could you try rephrasing or asking something else?"

    context_str = "\n\n---\n\n".join([f"Excerpt from S.I. {item['si_number']} ({item['year']}) - {item['title']}:\n{item['text']}\n(Source URL: {item['url']})" for item in relevant_si_texts])
    
    system_prompt = (
        "You are a helpful AI assistant specializing in Irish Statutory Instruments (S.I.s). "
        "You will be given a user's question and excerpts from relevant S.I. documents. "
        "Answer the user's question in a conversational tone, based *only* on the provided S.I. excerpts. "
        "Clearly cite the S.I. number and year for any information you use. "
        "If the provided excerpts do not contain enough information to answer the question, say so. "
        "Do not make up information or answer based on external knowledge."
    )
    
    user_message_content = f"User Question: {user_question}\n\nRelevant S.I. Excerpts:\n{context_str}"

    try:
        print("\nSending request to OpenAI ChatCompletion...")
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message_content}
            ],
            temperature=0.3, 
        )
        assistant_response = response.choices[0].message.content
        return assistant_response.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Sorry, I encountered an error trying to process your request with the AI model."


def main():
    if not load_data_and_model():
        return

    print("\nWelcome to the Irish S.I. Q&A system!")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        user_question = input("\nAsk your question: ").strip()
        if user_question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        if not user_question:
            continue

        print("Finding relevant S.I. chunks...")
        relevant_chunks = find_relevant_chunks(user_question)
        
        if not relevant_chunks:
            print("I couldn't find any S.I. chunks closely related to your question.")
            continue
            
        print(f"Found {len(relevant_chunks)} relevant chunk(s). Preparing response...")
        
        ai_response = get_openai_response(user_question, relevant_chunks)
        print(f"\nAssistant: {ai_response}")


if __name__ == '__main__':
    main()
