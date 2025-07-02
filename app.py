# app.py
# This version fixes the keyword search functionality by making the search more robust.
# All library requirements remain the same.

import os
import json
import re
import io
import numpy as np
import pandas as pd
import requests
import fitz  # PyMuPDF
import faiss
import camelot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import google.generativeai as genai

# --- Initialization & Model Loading ---
load_dotenv()
app = Flask(__name__)
CORS(app)

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Google Generative AI model configured.")
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key: print("Warning: SERPER_API_KEY not found. Web search will be disabled.")
except Exception as e:
    print(f"Error during initialization: {e}")
    llm = None

print("Loading ML models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
transcriber = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")
print("Models loaded.")

paper_storage = {}

# --- Core Helper Functions ---
def process_and_store_paper(pdf_url, title):
    global paper_storage
    if pdf_url in paper_storage: return True
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(pdf_url, headers=headers, timeout=60)
        response.raise_for_status()
        pdf_bytes = response.content
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        extracted_images = [{"page_num": p_num + 1, "bytes": doc.extract_image(img[0])["image"], "ext": doc.extract_image(img[0])["ext"]} for p_num, page in enumerate(doc) for img in page.get_images(full=True)]
        
        temp_pdf_path = f"temp_{hash(pdf_url)}.pdf"
        with open(temp_pdf_path, "wb") as f: f.write(pdf_bytes)
        tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='lattice')
        os.remove(temp_pdf_path)
        extracted_tables = [{"page_num": tbl.page, "table_index": i, "data": tbl.df.values.tolist()} for i, tbl in enumerate(tables)]
        
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        text_chunks = [cleaned_text[i:i + 1000] for i in range(0, len(cleaned_text), 800)]
        embeddings = embedding_model.encode(text_chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))
        paper_storage[pdf_url] = {"title": title, "chunks": text_chunks, "index": index, "images": extracted_images, "tables": extracted_tables}
        return True
    except Exception as e:
        print(f"Error processing {pdf_url}: {e}")
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        return False

def search_semantic_scholar_by_keyword(query, limit=5):
    """Searches Semantic Scholar for papers based on a keyword, with robust PDF finding."""
    try:
        url = 'https://api.semanticscholar.org/graph/v1/paper/search'
        # Fetch more results than needed, then filter locally for robustness
        params = { 
            'query': query, 
            'limit': limit * 3, 
            'fields': 'title,authors,year,openAccessPdf,externalIds' 
        }
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, params=params, headers=headers)
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
                    found_papers.append({'title': paper.get('title'), 'authors': [author['name'] for author in paper.get('authors', [])], 'year': paper.get('year'), 'pdfUrl': pdf_url})
                
                # Stop once we have found enough papers with PDFs
                if len(found_papers) >= limit:
                    break
        return found_papers
    except requests.exceptions.RequestException as e:
        print(f"Error searching Semantic Scholar: {e}")
        return []

def generate_answer_from_context(query, context_items, source_type):
    if not llm: return "The Language Model is not configured."
    context_str = ""
    if source_type == "DOCUMENTS":
        prompt_instruction = "Based *only* on the following context snippets from one or more research papers, provide a clear and concise answer to the user's question. Cite which paper your information comes from."
        for item in context_items:
            context_str += f"From paper '{item['title']}':\n\"...{item['chunk']}...\"\n\n"
    else: # Web search
        prompt_instruction = "Based on the following web search results, provide a clear and concise answer to the user's question."
        context_str = "\n---\n".join(context_items)
    prompt = f"{prompt_instruction}\n\nCONTEXT:\n---\n{context_str}\n---\n\nQUESTION: {query}\n\nANSWER:"
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error while generating the answer: {e}"

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
    if not serper_api_key: return ["Web search is disabled. No SERPER_API_KEY was found."]
    print(f"Performing live web search for: {query}")
    url, payload = "https://google.serper.dev/search", json.dumps({"q": query})
    headers = {'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json()
        if 'answerBox' in results and results['answerBox'].get('snippet'):
            return [results['answerBox']['snippet']]
        elif 'answerBox' in results and results['answerBox'].get('answer'):
             return [results['answerBox']['answer']]
        return [item.get('snippet', '') for item in results.get('organic', [])] or ["No relevant snippets found."]
    except requests.exceptions.RequestException as e:
        return [f"There was an error performing the web search: {e}"]

def describe_image(image_bytes, prompt):
    if not llm: return "Vision LLM not configured."
    try:
        img = Image.open(io.BytesIO(image_bytes))
        response = llm.generate_content([prompt, img])
        return response.text
    except Exception as e: return f"Vision LLM Error: {e}"

def summarize_table(table_data):
    if not llm: return "Text LLM not configured."
    try:
        df = pd.DataFrame(table_data)
        prompt = f"Please provide a concise summary of the key findings from the following data table extracted from a research paper:\n\n{df.to_markdown(index=False)}"
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e: return f"Error during table summarization: {e}"

def visualize_table_data(table_data):
    try:
        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        labels, values_str = df.iloc[:, 0].astype(str), df.iloc[:, 1]
        values = pd.to_numeric(values_str.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
        if len(labels) == 0 or len(values) == 0: return None
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, values)
        ax.set_title('Visualization of Table Data', fontsize=16)
        ax.set_ylabel(df.columns[1] or 'Values')
        ax.set_xlabel(df.columns[0] or 'Categories')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        print(f"Error during data visualization: {e}")
        return None

def format_papers_as_citations(papers):
    return "\n".join([f"- {(p.get('authors')[0] + ' et al.') if p.get('authors') else 'N/A'} ({p.get('year', 'N/A')}). *{p.get('title', 'No Title')}*" for p in papers])


# --- API Routes ---
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/api/process-paper', methods=['POST'])
def process_paper_endpoint():
    data = request.get_json()
    paper_url, title = data.get('url'), data.get('title', 'Untitled')
    if not paper_url: return jsonify({'error': 'URL is required'}), 400
    if not process_and_store_paper(paper_url, title): return jsonify({'error': 'Failed to process paper'}), 500
    paper_data = paper_storage.get(paper_url, {})
    image_info = [{"page": img["page_num"], "index": i} for i, img in enumerate(paper_data.get("images", []))]
    table_info = [{"page": tbl["page_num"], "index": tbl["table_index"]} for tbl in paper_data.get("tables", [])]
    return jsonify({'image_count': len(image_info), 'images_found': image_info, 'table_count': len(table_info), 'tables_found': table_info})

@app.route('/api/keyword-search', methods=['POST'])
def keyword_search_endpoint():
    data = request.get_json()
    query = data.get('query')
    if not query: return jsonify({'error': 'A search query is required.'}), 400
    papers = search_semantic_scholar_by_keyword(query)
    return jsonify(papers)

@app.route('/api/find-citations', methods=['POST'])
def find_citations_endpoint():
    data = request.get_json()
    topic = data.get('topic')
    if not topic: return jsonify({'error': 'A topic is required.'}), 400
    papers = search_semantic_scholar_by_keyword(topic, limit=10)
    if not papers: return jsonify({'answer': f"I couldn't find any citable papers with PDFs for the topic: '{topic}'."})
    citation_list = format_papers_as_citations(papers)
    return jsonify({'answer': f"Here are some relevant papers for the topic '{topic}':\n\n{citation_list}"})

@app.route('/api/global-search', methods=['POST'])
def global_search_endpoint():
    data = request.get_json()
    query = data.get('query')
    if not query: return jsonify({'error': 'Query is required'}), 400
    processed_titles = [p['title'] for p in paper_storage.values()]
    search_strategy = should_use_web_search(query, processed_titles)
    if search_strategy == "WEB_SEARCH":
        search_results = web_search_tool(query)
        answer = generate_answer_from_context(query, search_results, "WEB")
        return jsonify({'answer': f"(From Web Search) {answer}"})
    else: # DOCUMENT_SEARCH
        if not paper_storage: return jsonify({'answer': "There are no papers processed yet."})
        query_embedding = embedding_model.encode([query])
        all_relevant_chunks = []
        for url, paper_data in paper_storage.items():
            index = paper_data['index']
            distances, indices = index.search(np.array(query_embedding, dtype=np.float32), 2)
            for i in range(len(indices[0])):
                all_relevant_chunks.append({'title': paper_data['title'], 'chunk': paper_data['chunks'][indices[0][i]], 'distance': distances[0][i]})
        if not all_relevant_chunks: return jsonify({'answer': "I couldn't find relevant information in the processed papers."})
        all_relevant_chunks.sort(key=lambda x: x['distance'])
        answer = generate_answer_from_context(query, all_relevant_chunks[:4], "DOCUMENTS")
        return jsonify({'answer': answer})

@app.route('/api/get-image', methods=['GET'])
def get_image_endpoint():
    paper_url, image_index = request.args.get('url'), int(request.args.get('image_index', 0))
    if paper_url in paper_storage and 0 <= image_index < len(paper_storage[paper_url]['images']):
        img_data = paper_storage[paper_url]['images'][image_index]
        return send_file(io.BytesIO(img_data['bytes']), mimetype=f"image/{img_data['ext']}")
    return "Image not found", 404

@app.route('/api/describe-image', methods=['POST'])
def describe_image_endpoint():
    data = request.get_json()
    url, image_index = data.get('url'), data.get('image_index')
    if not all([url, image_index is not None]): return jsonify({'error': 'URL and image_index required'}), 400
    if url not in paper_storage: return jsonify({'error': 'Paper not processed'}), 404
    try:
        img_bytes = paper_storage[url]['images'][image_index]['bytes']
        description = describe_image(img_bytes, "Describe this image from a research paper in detail.")
        return jsonify({'description': description})
    except IndexError: return jsonify({'error': 'Image index out of bounds.'}), 404

@app.route('/api/summarize-table', methods=['POST'])
def summarize_table_endpoint():
    data = request.get_json()
    url, table_index = data.get('url'), data.get('table_index')
    if not all([url, table_index is not None]): return jsonify({'error': 'URL and table_index required'}), 400
    if url not in paper_storage: return jsonify({'error': 'Paper not processed'}), 404
    try:
        table_data = paper_storage[url]['tables'][table_index]['data']
        summary = summarize_table(table_data)
        return jsonify({'summary': summary})
    except IndexError: return jsonify({'error': 'Table index out of bounds.'}), 404

@app.route('/api/visualize-table', methods=['POST'])
def visualize_table_endpoint():
    data = request.get_json()
    url, table_index = data.get('url'), data.get('table_index')
    if not all([url, table_index is not None]): return jsonify({'error': 'URL and table_index required'}), 400
    if url not in paper_storage: return jsonify({'error': 'Paper not processed'}), 404
    try:
        table_data = paper_storage[url]['tables'][table_index]['data']
        chart_buffer = visualize_table_data(table_data)
        if chart_buffer is None: return jsonify({'error': 'Could not generate visualization for this table.'}), 500
        return send_file(chart_buffer, mimetype='image/png')
    except IndexError: return jsonify({'error': 'Table index out of bounds.'}), 404

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio_endpoint():
    if 'audio' not in request.files: return jsonify({'error': 'No audio file found'}), 400
    audio_file = request.files['audio']
    try:
        result = transcriber(audio_file.read())
        return jsonify({'transcription': result["text"]})
    except Exception as e:
        print(f"Transcription Error: {e}")
        return jsonify({'error': 'Failed to transcribe audio.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
