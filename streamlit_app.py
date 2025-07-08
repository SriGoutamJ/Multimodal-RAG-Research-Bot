# streamlit_app.py
# This file contains the complete Streamlit application with a corrected main loop.
# To run:
# 1. Ensure all previous dependencies are installed.
# 2. Install new dependencies: pip install streamlit streamlit-mic-recorder
# 3. Run from your terminal: streamlit run streamlit_app.py

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
import streamlit as st
from streamlit_mic_recorder import mic_recorder

from dotenv import load_dotenv
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Modal RAG Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Caching for Expensive Model Loading ---
@st.cache_resource
def load_models():
    """Loads all necessary models and configurations once, and caches them."""
    print("Loading models...")
    try:
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        serper_api_key = os.getenv("SERPER_API_KEY")
        
        if not google_api_key:
            st.error("GOOGLE_API_KEY not found in .env file. Please add it.")
            return None, None, None, None
        
        genai.configure(api_key=google_api_key)
        llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        transcriber = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")
        
        print("All models loaded successfully.")
        return llm, embedding_model, transcriber, serper_api_key
    except Exception as e:
        st.error(f"Error during model loading: {e}")
        return None, None, None, None

llm, embedding_model, transcriber, serper_api_key = load_models()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Find a paper to start, or ask me a general question."}]
if "paper_storage" not in st.session_state:
    st.session_state.paper_storage = {}
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "follow_up_questions" not in st.session_state:
    st.session_state.follow_up_questions = []

# --- Core Helper Functions (Backend Logic) ---

def process_and_store_paper(pdf_url, title):
    if pdf_url in st.session_state.paper_storage:
        st.sidebar.warning(f"'{title}' has already been processed.")
        return True
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, headers=headers, timeout=60)
        response.raise_for_status()
        pdf_bytes = response.content
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        
        full_text = "".join(page.get_text() for page in doc)
        images = [{"page_num": p_num + 1, "bytes": doc.extract_image(img[0])["image"], "ext": doc.extract_image(img[0])["ext"]} for p_num, page in enumerate(doc) for img in page.get_images(full=True)]
        
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
        
        st.session_state.paper_storage[pdf_url] = {"title": title, "chunks": text_chunks, "index": index, "images": images, "tables": extracted_tables}
        return True
    except Exception as e:
        st.sidebar.error(f"Error processing {title}: {e}")
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        return False

def search_semantic_scholar_by_keyword(query, limit=5):
    try:
        url = 'https://api.semanticscholar.org/graph/v1/paper/search'
        params = { 'query': query, 'limit': limit * 2, 'fields': 'title,authors,year,openAccessPdf,externalIds' }
        headers = {'User-Agent': 'Mozilla/5.0'}
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
                if len(found_papers) >= limit: break
        return found_papers
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def should_use_web_search(query, document_titles):
    if not llm: return "DOCUMENT_SEARCH"
    prompt = f"You are a routing agent. Your job is to decide the best way to answer a user's query.\nYou have access to the following documents:\n- {', '.join(document_titles) if document_titles else 'None'}\n\nThe user's query is: \"{query}\"\n\nBased on the query and the document titles, can the query likely be answered using ONLY the content of the provided documents, or does it require a real-time web search for more recent information, current events, or general knowledge?\n\nRespond with only \"DOCUMENT_SEARCH\" or \"WEB_SEARCH\"."
    try:
        response = llm.generate_content(prompt)
        decision = response.text.strip()
        return decision if decision in ["WEB_SEARCH", "DOCUMENT_SEARCH"] else "DOCUMENT_SEARCH"
    except Exception:
        return "DOCUMENT_SEARCH"

def web_search_tool(query):
    if not serper_api_key: return ["Web search is disabled."]
    url, payload = "https://google.serper.dev/search", json.dumps({"q": query})
    headers = {'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json()
        if 'answerBox' in results and results['answerBox'].get('snippet'):
            return [results['answerBox']['snippet']]
        return [item.get('snippet', '') for item in results.get('organic', [])] or ["No relevant snippets found."]
    except Exception as e:
        return [f"Web search failed: {e}"]

def generate_answer_from_context(query, context_items, source_type):
    if not llm: return "LLM not configured."
    context_str = ""
    if source_type == "DOCUMENTS":
        prompt_instruction = "Based *only* on the following context snippets from one or more research papers, provide a clear and concise answer to the user's question. Cite which paper your information comes from."
        for item in context_items:
            context_str += f"From paper '{item['title']}':\n\"...{item['chunk']}...\"\n\n"
    else:
        prompt_instruction = "Based on the following web search results, provide a clear and concise answer to the user's question."
        context_str = "\n---\n".join(context_items)
    prompt = f"{prompt_instruction}\n\nCONTEXT:\n---\n{context_str}\n---\n\nQUESTION: {query}\n\nANSWER:"
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error while generating the answer: {e}"

def describe_image(image_bytes):
    if not llm: return "Vision LLM not configured."
    try:
        img = Image.open(io.BytesIO(image_bytes))
        prompt = "Describe this image from a research paper in detail. What is it showing? What are the key takeaways?"
        response = llm.generate_content([prompt, img])
        return response.text
    except Exception as e: return f"Vision LLM Error: {e}"

def summarize_table(table_data):
    if not llm: return "Text LLM not configured."
    try:
        df = pd.DataFrame(table_data)
        prompt = f"Please provide a concise summary of the key findings from the following data table:\n\n{df.to_markdown(index=False)}"
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e: return f"Error during table summarization: {e}"

def format_papers_as_citations(papers):
    """Formats a list of paper dictionaries into a simple, citable list."""
    citation_list = []
    for paper in papers:
        authors = paper.get('authors', [])
        author_str = (authors[0] + " et al.") if authors else "N/A"
        year_str = paper.get('year', 'N/A')
        title_str = paper.get('title', 'No Title')
        citation_list.append(f"- {author_str} ({year_str}). *{title_str}*")
    return "\n".join(citation_list)

def generate_follow_up_questions(context_chunks):
    if not llm: return []
    context = "\n".join(context_chunks)
    prompt = f"Based on the following text snippets that were just used to answer a user's question, generate 3 insightful follow-up questions. Present them as a simple JSON list of strings. Example: [\"Can you elaborate on the methodology?\", \"How does this compare to previous work?\"]\n\nCONTEXT:\n---\n{context}\n---\n\nQUESTIONS (JSON list of 3 strings):"
    try:
        response = llm.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, Exception) as e:
        return []

# --- Streamlit UI Components ---

# Sidebar
with st.sidebar:
    st.title("📄 RAG Research Assistant")
    
    st.header("1. Find & Process Papers")
    search_tab, url_tab, tools_tab = st.tabs(["Keyword Search", "Direct URL", "Tools"])

    with search_tab:
        keyword = st.text_input("Search for papers by keyword", key="keyword_search_input")
        if st.button("Search"):
            if keyword:
                with st.spinner("Searching..."):
                    st.session_state.search_results = search_semantic_scholar_by_keyword(keyword)
            else:
                st.warning("Please enter a keyword.")
        
        if st.session_state.search_results:
            st.write("---")
            st.subheader("Search Results")
            for paper in st.session_state.search_results:
                with st.container(border=True):
                    st.markdown(f"**{paper['title']}** ({paper['year']})")
                    st.caption(f"Authors: {', '.join(paper['authors'])}")
                    if st.button("Process this Paper", key=paper['pdfUrl']):
                        with st.spinner(f"Processing '{paper['title']}'..."):
                            success = process_and_store_paper(paper['pdfUrl'], paper['title'])
                        if success:
                            st.toast(f"Successfully processed '{paper['title']}'!")
                            st.session_state.messages.append({"role": "assistant", "content": f"Successfully processed **{paper['title']}**. It's now available for analysis."})
                            st.rerun()

    with url_tab:
        url = st.text_input("Enter PDF URL", key="url_input")
        if st.button("Process URL"):
            if url:
                 with st.spinner(f"Processing URL..."):
                    success = process_and_store_paper(url, f"Paper from URL")
                 if success:
                    st.toast("Successfully processed paper from URL!")
                    st.session_state.messages.append({"role": "assistant", "content": "Successfully processed paper from URL. It's now available for analysis."})
                    st.rerun()
            else:
                st.warning("Please enter a URL.")

    with tools_tab:
        st.subheader("Citation Finder")
        topic = st.text_input("Enter a research topic")
        if st.button("Find Citations"):
            if topic:
                with st.spinner("Finding relevant citations..."):
                    papers = search_semantic_scholar_by_keyword(topic, limit=10)
                    if papers:
                        citations = format_papers_as_citations(papers)
                        st.session_state.messages.append({"role": "assistant", "content": f"Here are some relevant citations for **{topic}**:\n\n{citations}"})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I couldn't find any citable papers for '{topic}'."})
                st.rerun()
            else:
                st.warning("Please enter a topic.")

    st.divider()
    st.header("2. Analyze Processed Papers")
    if not st.session_state.paper_storage:
        st.info("No papers processed yet.")
    else:
        for url, paper_data in st.session_state.paper_storage.items():
            with st.expander(f"**{paper_data['title']}**"):
                st.caption("Image Analysis")
                for i, img_data in enumerate(paper_data['images']):
                    st.image(img_data['bytes'], caption=f"Image {i+1} (Page {img_data['page_num']})")
                    if st.button(f"Describe Image {i+1}", key=f"desc_{url}_{i}"):
                        with st.spinner("Analyzing image..."):
                            description = describe_image(img_data['bytes'])
                            st.session_state.messages.append({"role": "assistant", "content": f"**Analysis of Image from *{paper_data['title']}*:**\n\n{description}"})
                            st.rerun()
                
                st.caption("Table Analysis")
                for i, tbl_data in enumerate(paper_data['tables']):
                     st.dataframe(pd.DataFrame(tbl_data['data']))
                     if st.button(f"Summarize Table {i+1}", key=f"sum_tbl_{url}_{i}"):
                         with st.spinner("Summarizing table..."):
                            summary = summarize_table(tbl_data['data'])
                            st.session_state.messages.append({"role": "assistant", "content": f"**Summary of Table from *{paper_data['title']}*:**\n\n{summary}"})
                            st.rerun()

# Main chat interface
st.header("Chat with your Research Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle follow-up questions
if "follow_up_questions" in st.session_state and st.session_state.follow_up_questions:
    st.markdown("---")
    st.write("Suggested follow-ups:")
    cols = st.columns(len(st.session_state.follow_up_questions))
    for i, question in enumerate(st.session_state.follow_up_questions):
        if cols[i].button(question, key=f"follow_up_{i}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.follow_up_questions = [] # Clear follow-ups after one is clicked
            st.rerun()

# User input area
audio = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key='mic', just_once=True)

if audio:
    with st.spinner("Transcribing..."):
        audio_bytes = audio['bytes']
        transcription = transcriber(audio_bytes)
        st.session_state.messages.append({"role": "user", "content": transcription['text']})
        st.rerun()

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Main logic for generating responses, triggered by the last message being from the user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            last_prompt = st.session_state.messages[-1]["content"]
            processed_titles = [p['title'] for p in st.session_state.paper_storage.values()]
            search_strategy = should_use_web_search(last_prompt, processed_titles)
            
            context_for_followup = []
            if search_strategy == "WEB_SEARCH":
                st.info("Searching the web...")
                search_results = web_search_tool(last_prompt)
                response = generate_answer_from_context(last_prompt, search_results, "WEB")
                full_response = f"**(From Web Search)** {response}"
                context_for_followup = search_results
            else: # DOCUMENT_SEARCH
                if not st.session_state.paper_storage:
                    full_response = "I don't have any documents to search. Please process a paper first."
                else:
                    st.info("Searching processed documents...")
                    query_embedding = embedding_model.encode([last_prompt])
                    all_chunks = []
                    for url, paper_data in st.session_state.paper_storage.items():
                        index = paper_data['index']
                        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), 2)
                        for i in range(len(indices[0])):
                            all_chunks.append({'title': paper_data['title'], 'chunk': paper_data['chunks'][indices[0][i]], 'distance': distances[0][i]})
                    
                    if not all_chunks:
                        full_response = "I couldn't find relevant information in the processed papers."
                    else:
                        all_chunks.sort(key=lambda x: x['distance'])
                        top_chunks = all_chunks[:4]
                        full_response = generate_answer_from_context(last_prompt, top_chunks, "DOCUMENTS")
                        context_for_followup = [item['chunk'] for item in top_chunks]

            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            if context_for_followup:
                st.session_state.follow_up_questions = generate_follow_up_questions(context_for_followup)
            else:
                st.session_state.follow_up_questions = []
            
            st.rerun()
