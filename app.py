# app.py
# This version fixes the missing 'describe_image' function.
# All library requirements remain the same.

import os
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import requests
import fitz
import io
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
from transformers import pipeline
import json

# --- Initialization & Model Loading ---
load_dotenv()
app = Flask(__name__)
CORS(app)

try:
    # Load Google API Key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key: raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=google_api_key)
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Google Generative AI model configured.")

    # Load Serper API Key
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        print("Warning: SERPER_API_KEY not found. Web search will be disabled.")
    
except Exception as e:
    print(f"Error during initialization: {e}")
    llm = None

print("Loading ML models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
transcriber = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")
print("Models loaded.")

paper_storage = {}

# --- Core Processing & Helper Functions ---
def process_and_store_paper(pdf_url, title):
    global paper_storage
    if pdf_url in paper_storage: return True
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, headers=headers, timeout=60)
        response.raise_for_status()
        pdf_bytes = response.content
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        extracted_images = [{"page_num": p_num + 1, "bytes": base_image["image"], "ext": base_image["ext"]} for p_num, page in enumerate(doc) for img_index, img in enumerate(page.get_images(full=True)) for xref in [img[0]] for base_image in [doc.extract_image(xref)]]
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        text_chunks = [cleaned_text[i:i + 1000] for i in range(0, len(cleaned_text), 800)]
        embeddings = embedding_model.encode(text_chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))
        paper_storage[pdf_url] = {"title": title, "chunks": text_chunks, "index": index, "images": extracted_images}
        return True
    except Exception as e:
        print(f"Error processing {pdf_url}: {e}")
        return False

def search_semantic_scholar_by_keyword(query, limit=5):
    """Searches Semantic Scholar for papers based on a keyword."""
    try:
        url = 'https://api.semanticscholar.org/graph/v1/paper/search'
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,authors,year,openAccessPdf,externalIds',
            'openAccessPdf': 'true'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        
        found_papers = []
        if results.get("data"):
            for paper in results["data"]:
                pdf_url = None
                if paper.get('openAccessPdf') and paper['openAccessPdf'].get('url'):
                    pdf_url = paper['openAccessPdf']['url']
                elif paper.get('externalIds', {}).get('ArXiv'):
                    pdf_url = f"https://arxiv.org/pdf/{paper['externalIds']['ArXiv']}.pdf"
                
                if pdf_url:
                    found_papers.append({
                        'title': paper.get('title'),
                        'authors': [author['name'] for author in paper.get('authors', [])],
                        'year': paper.get('year'),
                        'pdfUrl': pdf_url
                    })
        return found_papers
    except requests.exceptions.RequestException as e:
        print(f"Error searching Semantic Scholar: {e}")
        return []

def generate_answer_from_context(query, context_items, source_type):
    if not llm: return "The Language Model is not configured."
    context_str = ""
    if source_type == "DOCUMENTS":
        for item in context_items:
            context_str += f"From paper '{item['title']}':\n\"...{item['chunk']}...\"\n\n"
        prompt_instruction = "Based *only* on the following context snippets from one or more research papers, provide a clear and concise answer to the user's question. Cite which paper your information comes from."
    else: # Web search
        context_str = "\n---\n".join(context_items)
        prompt_instruction = "Based on the following web search results, provide a clear and concise answer to the user's question."
    prompt = f"{prompt_instruction}\n\nCONTEXT:\n---\n{context_str}\n---\n\nQUESTION: {query}\n\nANSWER:"
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred during LLM generation: {e}")
        return "Sorry, I encountered an error while generating the answer."

def should_use_web_search(query, document_titles):
    if not llm: return "DOCUMENT_SEARCH"
    prompt = f"You are a routing agent. Your job is to decide the best way to answer a user's query.\nYou have access to the following documents:\n- {', '.join(document_titles) if document_titles else 'None'}\n\nThe user's query is: \"{query}\"\n\nBased on the query and the document titles, can the query likely be answered using ONLY the content of the provided documents, or does it require a real-time web search for more recent information, current events, or general knowledge?\n\nRespond with only \"DOCUMENT_SEARCH\" or \"WEB_SEARCH\"."
    try:
        response = llm.generate_content(prompt)
        decision = response.text.strip()
        print(f"Routing decision: {decision} for query: '{query}'")
        return decision if decision in ["WEB_SEARCH", "DOCUMENT_SEARCH"] else "DOCUMENT_SEARCH"
    except Exception as e:
        print(f"Router LLM error: {e}")
        return "DOCUMENT_SEARCH"

def web_search_tool(query):
    """Performs a live web search using the Serper API."""
    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key:
        return ["Web search is disabled. No SERPER_API_KEY was found."]
    print(f"Performing live web search for: {query}")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {'X-API-KEY': serper_key, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json()
        if 'answerBox' in results and results['answerBox'].get('snippet'):
            return [results['answerBox']['snippet']]
        elif 'answerBox' in results and results['answerBox'].get('answer'):
             return [results['answerBox']['answer']]
        snippets = [item.get('snippet', '') for item in results.get('organic', [])]
        return snippets if snippets else ["No relevant snippets found in the web search results."]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Serper API: {e}")
        return [f"There was an error performing the web search: {e}"]

# FIX: Re-added the missing describe_image function
def describe_image(image_bytes, prompt):
    """Analyzes an image using the vision model and a prompt."""
    if not llm: return "Vision LLM not configured."
    try:
        img = Image.open(io.BytesIO(image_bytes))
        response = llm.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"Vision LLM Error: {e}"


# --- Frontend Serving and API Routes ---
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/api/global-search', methods=['POST'])
def global_search_endpoint():
    data = request.get_json()
    query = data.get('query')
    if not query: return jsonify({'error': 'Query is required'}), 400
    processed_titles = [p['title'] for p in paper_storage.values()]
    search_strategy = should_use_web_search(query, processed_titles)
    if search_strategy == "WEB_SEARCH":
        search_results = web_search_tool(query)
        if not search_results:
            return jsonify({'answer': "I couldn't find any relevant information on the web."})
        answer = generate_answer_from_context(query, search_results, "WEB")
        return jsonify({'answer': f"(From Web Search) {answer}"})
    else: # DOCUMENT_SEARCH
        if not paper_storage:
            return jsonify({'answer': "There are no papers processed. Please find and process a paper first."})
        query_embedding = embedding_model.encode([query])
        all_relevant_chunks = []
        for url, paper_data in paper_storage.items():
            index = paper_data['index']
            distances, indices = index.search(np.array(query_embedding, dtype=np.float32), 2)
            for i in range(len(indices[0])):
                all_relevant_chunks.append({'title': paper_data['title'], 'chunk': paper_data['chunks'][indices[0][i]], 'distance': distances[0][i]})
        if not all_relevant_chunks:
            return jsonify({'answer': "I couldn't find relevant information in the processed papers to answer that."})
        all_relevant_chunks.sort(key=lambda x: x['distance'])
        top_chunks_for_llm = all_relevant_chunks[:4]
        answer = generate_answer_from_context(query, top_chunks_for_llm, "DOCUMENTS")
        return jsonify({'answer': answer})

@app.route('/api/process-paper', methods=['POST'])
def process_paper_endpoint():
    data = request.get_json()
    paper_url, title = data.get('url'), data.get('title', 'Untitled')
    if not paper_url: return jsonify({'error': 'URL is required'}), 400
    if not process_and_store_paper(paper_url, title): return jsonify({'error': 'Failed to process paper'}), 500
    paper_data = paper_storage.get(paper_url, {})
    image_info = [{"page": img["page_num"], "index": i} for i, img in enumerate(paper_data.get("images", []))]
    return jsonify({'image_count': len(image_info), 'images_found': image_info})
    
@app.route('/api/keyword-search', methods=['POST'])
def keyword_search_endpoint():
    data = request.get_json()
    query = data.get('query')
    if not query: return jsonify({'error': 'A search query is required.'}), 400
    papers = search_semantic_scholar_by_keyword(query)
    return jsonify(papers)

@app.route('/api/get-image', methods=['GET'])
def get_image_endpoint():
    paper_url = request.args.get('url')
    image_index = int(request.args.get('image_index', 0))
    if paper_url in paper_storage and 0 <= image_index < len(paper_storage[paper_url]['images']):
        img_data = paper_storage[paper_url]['images'][image_index]
        return send_file(io.BytesIO(img_data['bytes']), mimetype=f"image/{img_data['ext']}")
    return "Image not found", 404

@app.route('/api/describe-image', methods=['POST'])
def describe_image_endpoint():
    data = request.get_json()
    paper_url, image_index = data.get('url'), data.get('image_index')
    if not all([paper_url, image_index is not None]): return jsonify({'error': 'URL and image_index required'}), 400
    if paper_url not in paper_storage: return jsonify({'error': 'Paper not processed'}), 404
    try:
        img_bytes = paper_storage[paper_url]['images'][image_index]['bytes']
        prompt = "Describe this image from a research paper in detail."
        description = describe_image(img_bytes, prompt)
        return jsonify({'description': description})
    except IndexError:
        return jsonify({'error': 'Image index out of bounds'}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5000)
