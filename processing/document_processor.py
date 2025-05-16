import os
import json
import re
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CRAWLED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'crawled_docs')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_docs')
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

def load_s_i_documents(data_dir=CRAWLED_DATA_DIR):
    """Loads all S.I. JSON documents from the specified directory."""
    documents = []
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        return documents

    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    doc_content = json.load(f)
                    # Assuming the main text is under a 'text_content' key
                    # and we also want to keep metadata like title and S.I. number
                    
                    year_val = doc_content.get('year')
                    si_number_simple_val = doc_content.get('si_number_simple')
                    
                    if year_val is None and si_number_simple_val:
                        parts = si_number_simple_val.split('/')
                        if len(parts) == 2 and parts[1].isdigit():
                            year_val = parts[1]
                            
                    documents.append({
                        'si_number': si_number_simple_val,
                        'title': doc_content.get('title'),
                        'text': doc_content.get('text_content'),
                        'url': doc_content.get('url'),
                        'year': year_val, # Derived or original year
                        'original_filename': filename
                    })
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {filepath}")
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
    return documents

def chunk_text_by_paragraph(text, min_chunk_size=50):
    """Chunks text by paragraphs (separated by double newlines).
       Filters out very small chunks.
    """
    if not text:
        return []
    paragraphs = re.split(r'\n\s*\n', text) # Split by one or more newlines, possibly with whitespace
    
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If current_chunk is too small, append to it, otherwise start a new chunk
        if len(current_chunk) < min_chunk_size and current_chunk:
            current_chunk += "\n\n" + para
        else:
            if current_chunk: # Add the completed chunk
                 chunks.append(current_chunk.strip())
            current_chunk = para
            
    if current_chunk: # Add the last remaining chunk
        chunks.append(current_chunk.strip())
        
    # Further filter to ensure all chunks meet min_chunk_size (approximately)
    # This handles cases where individual paragraphs are very short
    # and the merging logic above isn't sufficient
    final_chunks = []
    buffer_chunk = ""
    for chunk in chunks:
        if len(buffer_chunk) == 0:
            buffer_chunk = chunk
        elif len(buffer_chunk) < min_chunk_size:
            buffer_chunk += "\n\n" + chunk
        else:
            final_chunks.append(buffer_chunk)
            buffer_chunk = chunk
    if buffer_chunk:
        final_chunks.append(buffer_chunk) # Add the very last chunk
        
    return [c for c in final_chunks if len(c) >= min_chunk_size or len(c) > 0 and not final_chunks]


def process_documents():
    """Main function to load, chunk, and embed documents."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set. Please set it to use OpenAI embeddings.")
        return
    try:
        client = openai.OpenAI(api_key=api_key)
        print(f"OpenAI client initialized. Using model: {OPENAI_EMBEDDING_MODEL}")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return

    documents = load_s_i_documents()
    if not documents:
        print("No documents found or loaded. Exiting.")
        return

    all_chunked_data = []

    for i, doc in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}: {doc.get('original_filename')}")
        if not doc.get('text'):
            print(f"  Skipping document {doc.get('original_filename')} due to missing text.")
            continue

        chunks = chunk_text_by_paragraph(doc['text'])
        
        if not chunks:
            print(f"  No chunks generated for document {doc.get('original_filename')}. Text length: {len(doc['text'])}")
            # Store even if no chunks, to signify it was processed
            # Or decide to skip based on requirements

        for chunk_id, chunk_text in enumerate(chunks):
            all_chunked_data.append({
                'si_number': doc.get('si_number'),
                'title': doc.get('title'),
                'url': doc.get('url'),
                'year': doc.get('year'),
                'original_filename': doc.get('original_filename'),
                'chunk_id': f"{doc.get('si_number', 'unknown_si')}_{chunk_id}",
                'text': chunk_text
                # Embedding will be added later after all chunks are collected for potential batching
            })
        
        # For now, let's print the number of chunks for the first few documents
        if i < 5:
            print(f"  Generated {len(chunks)} chunks for {doc.get('original_filename')}.")

    # Save the chunked data to a new JSON file
    output_filepath = os.path.join(OUTPUT_DIR, 'all_chunked_data_with_embeddings.json') # Changed filename
    
    # Generate embeddings for all chunks
    if all_chunked_data:
        print(f"\nGenerating embeddings for {len(all_chunked_data)} chunks using OpenAI...")
        texts_to_encode = [chunk['text'] for chunk in all_chunked_data]
        all_embeddings = []
        batch_size = 200 # Define a batch size
        num_batches = (len(texts_to_encode) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(texts_to_encode))
            batch_texts = texts_to_encode[start_index:end_index]
            
            if not batch_texts: # Should not happen with correct batch logic but as a safeguard
                continue

            print(f"  Processing batch {i+1}/{num_batches} ({len(batch_texts)} texts)")
            try:
                response = client.embeddings.create(
                    input=batch_texts,
                    model=OPENAI_EMBEDDING_MODEL
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating OpenAI embeddings for batch {i+1}: {e}")
                print(f"  Skipping this batch. {len(batch_texts)} embeddings will be missing.")
                # Add None or a placeholder for failed batch embeddings to maintain length alignment
                all_embeddings.extend([None] * len(batch_texts)) 
        
        # Add embeddings to each chunk dictionary if lengths match
        if len(all_embeddings) == len(all_chunked_data):
            embeddings_added_count = 0
            for i, chunk_data in enumerate(all_chunked_data):
                if all_embeddings[i] is not None:
                    chunk_data['embedding'] = all_embeddings[i]
                    embeddings_added_count += 1
            if embeddings_added_count == len(all_chunked_data):
                print("Embeddings generated and added to all chunks successfully.")
            else:
                print(f"Embeddings added to {embeddings_added_count}/{len(all_chunked_data)} chunks. Some batches may have failed.")
        else:
            print(f"Error: Mismatch in number of total embeddings generated ({len(all_embeddings)}) and chunks ({len(all_chunked_data)}).")
            print("Skipping embedding addition to prevent data corruption.")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            json.dump(all_chunked_data, f_out, indent=2, ensure_ascii=False)
        print(f"Successfully saved all chunked data with embeddings to {output_filepath}")
    except Exception as e:
        print(f"Error saving chunked data with embeddings: {e}")

if __name__ == '__main__':
    process_documents()
