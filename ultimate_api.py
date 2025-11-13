import os
import json
import re
import urllib.parse
import logging
import html
import inspect
import requests
from datetime import datetime, timezone, date
from typing import List, Dict, Optional, Tuple, Union, Set
from pathlib import Path
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from traceback import format_exc
from pydantic import BaseModel, ValidationError

# Use pydantic_settings for environment variable loading
from pydantic_settings import BaseSettings
from collections import defaultdict, Counter 

# --- Configuration (Loaded from Environment/Settings) -----------------------

class Settings(BaseSettings):
    """
    Configuration loaded from environment variables (e.g., OPENAI_APIKEY).
    It defaults chunks_path for robustness.
    """
    openai_apikey: str
    openai_model: str 
    chunks_path: str = "processed/processed_chunks.json" 

settings = Settings()
CHUNKS_PATH = settings.chunks_path
PROMPT_CHARS_BUDGET = int(os.getenv("PROMPT_CHARS_BUDGET", "24000"))
MAX_HITS_IN_PROMPT = 12
MAX_SNIPPET_CHARS = 800

# --- Pydantic Models (Consolidated) -----------------------------------------

class SourceItem(BaseModel):
    """Model for a single content source."""
    title: str
    url: str
    snippet: Optional[str] = None 

class AnswerResponse(BaseModel):
    """Model for the final API response."""
    answer: str
    sources: List[SourceItem] = []

class QuestionRequest(BaseModel):
    """Model for the request body of the /ask endpoint."""
    question: str
    language: Optional[str] = "de"
    max_sources: Optional[int] = 3

# --- Logging Setup ----------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Datums-Parsing (de) ----------------------------------------------------

# z.B. "11.11.2025", "11.11.25"
DMY_DOTTED_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b")
# z.B. "25. Nov 2025", "25. November 25", "25. November"
DMY_TEXT_RE = re.compile(
    r"\b(\d{1,2})\.\s*([A-Za-zäöüÄÖÜ]+)\s*(\d{2,4})?\b",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"\b(\d{1,2})[:.](\d{2})\b")

MONTHS_DE = {
    "jan": 1, "januar": 1, "feb": 2, "februar": 2, "mär": 3, "maerz": 3, 
    "märz": 3, "mar": 3, "apr": 4, "april": 4, "mai": 5, "jun": 6, "juni": 6,
    "jul": 7, "juli": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, 
    "september": 9, "okt": 10, "oktober": 10, "nov": 11, "november": 11,
    "dez": 12, "dezember": 12,
}

# --- Global Constants & Initialization --------------------------------------

from openai import OpenAI
IMPULS_URL = "https://dlh.zh.ch/home/impuls-workshops" 

openai_client = OpenAI(api_key=settings.openai_apikey)
app = FastAPI(title="DLH OpenAI API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://perino.info"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --- Utility Functions ------------------------------------------------------

def safe_add_lists(a, b):
    return ensure_list(a) + ensure_list(b)

def ensure_list(val):
    """Convert None to [], lists unchanged, single values/objects to [value], and AnswerResponse to list of its .sources."""
    if val is None:
        return []
    if hasattr(val, "sources"):
        return ensure_list(val.sources)
    if isinstance(val, list):
        return val
    return [val]

def call_openai(system_prompt, user_prompt, max_tokens=1200):
    """
    Calls the OpenAI API. Uses max_completion_tokens (required by models like gpt-5) 
    and includes a basic exception handler for models requiring max_tokens (standard).
    """
    try:
        # 1. Attempt using max_completion_tokens (as requested by your model error)
        response = openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=max_tokens, 
            stream=False,
        )
    except Exception as e:
        error_msg = str(e)
        # Check for the specific unsupported parameter error 
        if 'unsupported parameter' in error_msg.lower() and 'max_tokens' in error_msg.lower():
            logger.warning(f"Model {settings.openai_model} rejected max_completion_tokens. Falling back to max_tokens.")
            # 2. Fallback to max_tokens (standard for most official OpenAI models)
            try:
                response = openai_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens, 
                    stream=False,
                )
            except Exception as e_fallback:
                logger.error(f"OpenAI API ERROR (Fallback failed): {repr(e_fallback)}\n{format_exc()}")
                return "<p>Fehler bei der KI-Antwort. Bitte später erneut versuchen.</p>"
        else:
            # Re-raise if it's a different, unrecoverable error
            logger.error(f"OpenAI API ERROR (unhandled): {repr(e)}\n{format_exc()}")
            return "<p>Fehler bei der KI-Antwort. Bitte später erneut versuchen.</p>"

    # Process response
    result = response.choices[0].message.content.strip()
    logger.info(f"OpenAI returned: {repr(result)[:400]}")
    if not result.strip():
        result = "<p>Leider konnte keine Antwort generiert werden.</p>"
    return result


def summarize_long_text(text, max_length=180):
    # If short, just return as is
    if not text or len(text) < 200:
        return text
    prompt = f"Fasse den folgenden Text in zwei Sätzen zusammen:\n{text}"
    try:
        response = openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_length,
            stream=False
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logger.error(f"OpenAI Summarization error: {repr(e)}")
        return text[:max_length]

def render_workshops_timeline_html(events: List[Dict], title: str = "Kommende Impuls-Workshops") -> str:
    """
    Erzeugt eine einfache HTML-Timeline mit Datum + klickbarem Titel.
    """
    if not events:
        html_str = (
            "<section class='dlh-answer'>"
            f"<p>Keine {title.replace('(aus KB)', '').replace('(Live)', '').strip()} gefunden.</p>"
            "<h3>Quellen</h3>"
            "<ul class='sources'>"
            f"<li><a href='{IMPULS_URL}' target='_blank'>Impuls-Workshop-Übersicht</a></li>"
            "</ul></section>"
        )
        logger.info(f"Returning answer: {repr(html_str)[:400]}")
        return html_str

    items_html = []
    for e in events:
        d = _event_to_date(e)
        date_str = d.strftime("%d.%m.%Y") if d else ""
        t = e.get("title", "Ohne Titel")
        url = e.get("url") or IMPULS_URL
        place = e.get("place") or ""
        meta = f"<div class='meta'>{place}</div>" if place else ""
        items_html.append(
            f"<li><time>{date_str}</time> "
            f"<a href='{url}' target='_blank' rel='noopener noreferrer'>{t}</a>{meta}</li>"
        )

    # Use title directly from the function argument, which includes (aus KB) or (Live)
    html_str = (
        "<section class='dlh-answer'>"
        f"<p>{title}:</p>"
        "<ol class='timeline'>" + "".join(items_html) + "</ol>"
        "<h3>Quellen</h3>"
        "<ul class='sources'>"
        f"<li><a href='{IMPULS_URL}' target='_blank'>Impuls-Workshop-Übersicht</a></li>"
        "</ul></section>"
    )
    logger.info(f"Returning answer: {repr(html_str)[:400]}")
    return html_str


def render_structured_cards_html(items: List[Dict], title: str, source_url: str) -> str:
    """
    Generiert die HTML-Struktur für Projekte/Angebote (Karten-Layout) aus Python-Daten.
    """
    if not items:
        return f"<p>Leider wurden keine passenden {title.replace('(aus KB)', '').strip()} gefunden.</p>"

    cards_html = []
    for item in items:
        # Ensure title and URL exist and create summary from content if available
        item_title = item.get("title", "Unbekanntes Element")
        item_url = item.get("url", "#")
        item_desc = item.get("description") or item.get("content", "")
        
        # Use existing summary if available, otherwise generate one
        item_snippet = item.get("snippet") or summarize_long_text(item_desc) 
        
        cards_html.append(
            f"<article class='card'>"
            f"<h4><a href='{item_url}' target='_blank'>{item_title}</a></h4>"
            f"<p>{item_snippet}</p>"
            f"</article>"
        )

    html_str = (
        f"<section class='dlh-answer'>"
        f"<p>{title} (aus KB):</p>"
        f"<div class='cards'>" + "".join(cards_html) + "</div>"
        f"<h3>Quellen</h3>"
        f"<ul class='sources'>"
        f"<li><a href='{source_url}' target='_blank'>DLH Wissensbasis</a></li>"
        "</ul></section>"
    )
    return html_str


def extract_innovationsfonds_projects(chunks: List[Dict]) -> List[Dict]:
    """Extrahiert alle Projekte mit Metadaten aus den relevanten Chunks."""
    projects = []
    for ch in chunks:
        # Check if the chunk contains project-specific metadata/content
        if "innovationsfonds" in ch.get('url', '').lower() or ch.get('metadata', {}).get('chunk_type') == 'innovationsfonds_project':
            # Extract key fields. Content should be shortened for the card description.
            title = ch.get('title')
            url = ch.get('url', ch.get('metadata', {}).get('source', '#'))
            content = ch.get('content', '')
            
            # Use specific project metadata keys if available (common in your chunks)
            project_title = ch.get('metadata', {}).get('title', title)
            
            # The project summary is often the content field itself or a separate snippet/description
            project_description = ch.get('content', '')
            
            if project_title:
                projects.append({
                    "title": project_title,
                    "url": url,
                    "description": project_description # Full content to be summarized by render function
                })
    
    # Simple deduplication based on title/url pair
    seen_keys = set()
    deduplicated = []
    for p in projects:
        key = (p['title'], p['url'])
        if key not in seen_keys:
            deduplicated.append(p)
            seen_keys.add(key)
            
    return deduplicated


def extract_fobizz_resources(chunks: List[Dict]) -> List[Dict]:
    """Extrahiert allgemeine Fobizz-Angebote aus den relevanten Chunks."""
    fobizz_items = []
    for ch in chunks:
        if "fobizz" in ch.get('url', '').lower() or "fobizz" in ch.get('title', '').lower() or "fobizz" in ch.get('content', '').lower():
            title = ch.get('title')
            url = ch.get('url', ch.get('metadata', {}).get('source', '#'))
            content = ch.get('content', '')
            
            if title:
                fobizz_items.append({
                    "title": title,
                    "url": url,
                    "description": content
                })
    # Simple deduplication
    seen_keys = set()
    deduplicated = []
    for p in fobizz_items:
        key = (p['title'], p['url'])
        if key not in seen_keys:
            deduplicated.append(p)
            seen_keys.add(key)
            
    return deduplicated


def _event_to_date(e: Dict) -> Optional[datetime]:
    """
    Hilfsfunktion für die Workshop-Intentlogik:
    nimmt ein Event-Dict und liefert ein datetime-Objekt.
    """
    if not isinstance(e, dict):
        return None

    d = e.get("date")
    if isinstance(d, datetime):
        return d

    txt = e.get("when") or e.get("title") or ""
    return parse_de_date(txt)

# --- Date Parsing Helpers ---------------------------------------------------

def coerce_year(y, refyear):
    if not y:
        return refyear
    y = y.strip()
    if len(y) == 2:
        y = "20" + y
    try:
        return int(y)
    except Exception:
        return refyear

def parse_de_date(text: str, ref_date: Optional[datetime] = None) -> Optional[datetime]:
    """Versucht ein Datum (und ggf. Uhrzeit) aus deutschem Text zu extrahieren."""
    if not text:
        return None
    t = text.strip()
    rd = ref_date or datetime.now(timezone.utc)
    ref_year = rd.year

    # 1) 11.11.2025
    m = DMY_DOTTED_RE.search(t)
    hh, mm = None, None
    tm = TIME_RE.search(t)
    if tm:
        hh, mm = int(tm.group(1)), int(tm.group(2))
    if m:
        d = int(m.group(1))
        mth = int(m.group(2))
        y = coerce_year(m.group(3), ref_year)
        try:
            base = datetime(y, mth, d, tzinfo=timezone.utc)
            if hh is not None:
                base = base.replace(hour=hh, minute=mm or 0)
            return base
        except Exception:
            pass

    # 2) 25. Nov 2025 / 25. November (Jahr optional)
    m = DMY_TEXT_RE.search(t)
    if m:
        d = int(m.group(1))
        month_word = (m.group(2) or "").strip().lower()
        # Umlaute vereinheitlichen
        month_word = (month_word
                      .replace("ä", "ae")
                      .replace("ö", "oe")
                      .replace("ü", "ue"))
        mth = MONTHS_DE.get(month_word)
        y = coerce_year(m.group(3), ref_year)
        if mth:
            try:
                base = datetime(y, mth, d, tzinfo=timezone.utc)
                if hh is not None:
                    base = base.replace(hour=hh, minute=mm or 0)
                return base
            except Exception:
                pass

    return None

# --- Main API Endpoint ------------------------------------------------------

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    try:
        try:
            # 1. Perform retrieval from chunks regardless of intent
            ranked = get_ranked_with_sitemap(req.question, max_items=req.max_sources or 12)
        except Exception as e:
            logger.warning(f"Sitemap or advanced search failed: {repr(e)}. Falling back to advanced_search.")
            ranked = advanced_search(req.question, max_items=req.max_sources or 12)
        
        print("Y ranked types:", [type(x).__name__ for x in ranked[:5]])
        q_low = (req.question or "").lower()


        # ---- 1. STRUCTURED HANDLER (Innovationsfonds / Fobizz / Vernetzung) ----

        # 1a. Innovationsfonds Projects
        if any(k in q_low for k in ["innovationsfonds", "projekte", "innovation"]):
            projects = extract_innovationsfonds_projects(ranked)
            if projects:
                html_answer = render_structured_cards_html(
                    projects, 
                    title="DLH Innovationsfonds Projekte", 
                    source_url="https://dlh.zh.ch/home/innovationsfonds/projektvorstellungen/uebersicht"
                )
                return AnswerResponse(
                    answer=html_answer,
                    sources=[SourceItem(title=p['title'], url=p['url']) for p in projects[:req.max_sources or 4]]
                )
        
        # 1b. Fobizz Resources
        if any(k in q_low for k in ["fobizz", "tool", "sprechstunde", "kurs", "weiterbildung"]):
            fobizz_items = extract_fobizz_resources(ranked)
            if fobizz_items:
                html_answer = render_structured_cards_html(
                    fobizz_items, 
                    title="DLH Fobizz Angebote", 
                    source_url="https://dlh.zh.ch/home/wb-kompass/wb-angebote/1007-wb-plattformen"
                )
                return AnswerResponse(
                    answer=html_answer,
                    sources=[SourceItem(title=p['title'], url=p['url']) for p in fobizz_items[:req.max_sources or 4]]
                )

        # ---- 2. WORKSHOP INTENT (Time-sensitive, requires custom filtering/sorting) ---

        if any(k in q_low for k in ["impuls", "workshop", "workshops", "termine"]):
            
            # Intent checking functions (re-defined locally as they are complex)
            def _norm(s: str) -> str:
                return (s.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss"))
            qn = _norm(req.question or "")
            want_past = any(k in qn for k in ["gab es", "waren", "vergangenen", "bisherigen", "im jahr", "letzten", "fruehere", "frühere", "bisher"])
            want_next = any(k in qn for k in ["naechste", "nächste", "der naechste", "der nächste", "als naechstes", "als nächstes", "nur der naechste", "nur der nächste", "naechstes", "nächstes"])
            yr = None
            m = re.search(r"(?:jahr|jahrgang|seit)\s*(20\d{2})", qn)
            if m:
                try: yr = int(m.group(1))
                except ValueError: yr = None
            
            # --- CHUNK ANALYSIS (Step 1 & 2: Prioritize Chunks) ---
            chunk_events = []
            for ch in ranked:
                # Extract event data from chunk if it looks like an event
                u = ch.get('url', ch.get('metadata', {}).get('source', ''))
                
                # Check for relevance: must be related to events or workshops
                is_relevant_event = any(k in u.lower() for k in ["impuls-workshops", "termine", "events", "aktuell"])
                
                if is_relevant_event:
                    # Chunks often contain multiple dates/events in metadata. Extract all.
                    d_meta = ch.get('metadata', {}).get('dates', [])
                    
                    for d_str in d_meta:
                        dt_obj = parse_de_date(d_str)
                        
                        if dt_obj:
                             # Append the chunk/date pair as a single event item
                             # NOTE: We use the raw date string in the title to help differentiate items originating from the same page
                            chunk_events.append({
                                "date": dt_obj, 
                                "title": ch.get("title", "(Ohne Titel)") + f" ({d_str})", 
                                "url": u, 
                                "_d": dt_obj 
                            })

            if chunk_events:
                # Proceed with chunk-based filtering/rendering
                today = datetime.now(timezone.utc).date()
                future_chunk_events = [e for e in chunk_events if e["_d"].date() >= today]
                
                # Apply filters based on intent
                if want_next:
                    events_to_show = sorted(future_chunk_events, key=lambda x: x["_d"])[:1]
                    title_suffix = " (aus KB)"
                elif want_past:
                    past_chunk_events = [e for e in chunk_events if e["_d"].date() < today]
                    if yr: past_chunk_events = [e for e in past_chunk_events if e["_d"].year == yr]
                    events_to_show = sorted(past_chunk_events, key=lambda x: x["_d"], reverse=True)
                    title_suffix = " (aus KB)" + (f" {yr}" if yr else "")
                else: # Default: all future events from chunks
                    events_to_show = sorted(future_chunk_events, key=lambda x: x["_d"])
                    title_suffix = " (aus KB)"
                    
                html = render_workshops_timeline_html(
                    events_to_show, 
                    title=("Nächster Impuls-Workshop" if want_next else "Kommende Impuls-Workshops") + title_suffix
                )

                # Return result based on chunks (Prioritization)
                return AnswerResponse(
                    answer=html,
                    sources=[SourceItem(title=e['title'], url=e['url']) for e in events_to_show if 'title' in e and 'url' in e] 
                )

            # 4. Fallback to LIVE SCRAPER if CHUNKS yield nothing relevant

            events = fetch_live_impuls_workshops()
            norm_events = []
            for e in events:
                d = _event_to_date(e)
                if d:
                    ee = dict(e)
                    ee["_d"] = d
                    norm_events.append(ee)

            today = datetime.now(timezone.utc).date()
            future = [e for e in norm_events if e["_d"].date() >= today]
            past = [e for e in norm_events if e["_d"].date() < today]

            # Re-apply filtering/sorting logic on live data
            if want_next:
                future_sorted = sorted(future, key=lambda x: x["_d"])
                events_to_show = future_sorted[:1]
                html = render_workshops_timeline_html(events_to_show, title="Nächster Impuls-Workshop (Live)")
            elif want_past:
                if yr: past = [e for e in past if e["_d"].year == yr]
                past_sorted = sorted(past, key=lambda x: x["_d"], reverse=True)
                html = render_workshops_timeline_html(past_sorted, title="Vergangene Impuls-Workshops (Live)" + (f" {yr}" if yr else ""))
            else:
                future_sorted = sorted(future, key=lambda x: x["_d"])
                html = render_workshops_timeline_html(future_sorted, title="Kommende Impuls-Workshops (Live)")
            
            # Return result based on live pages 
            return AnswerResponse(
                answer=html,
                sources=[SourceItem(title="Impuls-Workshop-Übersicht", url=IMPULS_URL)],
            )
            
        # ---- Ende Workshop-Sonderfall
        
        # ---- 3. DEFAULT LLM/RAG BRANCH (General Concepts, Definitions, etc.) ----
        
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(req.question, ranked)
        answer_html = call_openai(system_prompt, user_prompt, max_tokens=1200)
        answer_html = ensure_clickable_links(answer_html)
        
        # Quellenliste für die AnswerResponse erzeugen
        sources = build_sources(ranked, limit=req.max_sources or 4)
        
        if not answer_html.strip():
             answer_html = "<p>Leider konnte keine Antwort generiert werden.</p>" # Fallback
             
        logger.info(f"Returning answer: {repr(answer_html)[:400]}")
        response = AnswerResponse(answer=answer_html, sources=sources)
        return response

    except Exception as e:
        msg = "Entschuldigung, es gab einen technischen Fehler. Bitte versuchen Sie es später erneut."
        logger.error(f"ERROR /ask: %s\n{format_exc()}", repr(e))
        logger.info(f"Returning answer: {repr(msg)[:400]}")
        return AnswerResponse(answer=msg, sources=[])

# --- Knowledge Base Loading and Indexing (The rest of the file) ------------------------------------

def load_chunks(path: str) -> List[Dict]:
    """Load the knowledge base .json or .jsonl robustly."""
    out = []
    p = Path(path)
    if not p.exists():
        logger.warning(f"⚠️ KB not found at {p.resolve()}")
        return []
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    logger.debug(f"Skipping invalid JSON line in {path}")
                    continue
        return out
    else:
        try:
            return json.load(p.open("r", encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load chunks from {path}: {e}")
            return []

# Load chunks immediately after settings and utility functions
CHUNKS: List[Dict] = load_chunks(CHUNKS_PATH)
CHUNKS_COUNT = len(CHUNKS)
logger.info(f"✅ Loaded {CHUNKS_COUNT} chunks from {CHUNKS_PATH}")


def index_chunks_by_subject(chunks):
    """Returns: subject -> [chunk ids]"""
    idx = defaultdict(list)
    for i, ch in enumerate(chunks):
        subject = ch.get("subject")
        if subject:
            idx[subject.lower()].append(i)
    return idx

SUBJECTINDEX = index_chunks_by_subject(CHUNKS)

def extract_terms(query: str) -> Set[str]:
    """Simple term extraction for keyword indexing."""
    query = query.lower()
    # Simple tokenization: split by non-alphanumeric, filter short/common words
    tokens = re.split(r"[^a-zäöüß0-9]+", query)
    return {t for t in tokens if len(t) > 2 and t not in ["der", "die", "das", "und", "oder"]}

def filter_chunks_by_section(chunks, section):
    """Return all chunks with given section label."""
    return [ch for ch in chunks if ch.get("section") == section]

def get_chunks_by_tag(tag):
    """Return all chunks with given tag present in tags array/string."""
    tag_lower = tag.lower()
    return [ch for ch in CHUNKS if tag_lower in str(ch.get("tags", [])).lower()]


def get_subject_faq(subject, max_items=10):
    # Return all FAQ chunks for a single subject/tag
    idxs = SUBJECTINDEX.get(subject.lower(), [])
    return [CHUNKS[i] for i in idxs][:max_items]

def build_faq_list_html(subject, max_length=1200):
    faqs = get_subject_faq(subject)
    arts = []
    for faq in faqs:
        title = faq.get("title", "(keine Überschrift)")
        url = faq.get("url", "#")
        snippet = faq.get("snippet", "")
        arts.append(f"<article><h4><a href='{url}' target='_blank'>{title}</a></h4><p>{snippet}</p></article>")
        if sum(len(a) for a in arts) > max_length:
            break
    html = "<div class='faq-list'>" + "\n".join(filter(None, ensure_list(arts))) + "</div>" if arts else "<p>Keine FAQs gefunden.</p>"
    logger.info(f"Returning answer: {repr(html)[:400]}")
    return html

def index_chunks_by_keywords(chunks):
    keywordindex = {}
    for i, ch in enumerate(chunks):
        keywords = ch.get("keywords", [])
        for kw in keywords:
            keywordindex.setdefault(kw.lower(), set()).add(i)
    return keywordindex

KEYWORDINDEX = index_chunks_by_keywords(CHUNKS)

def advanced_search(query, max_items=12):
    # Score chunks based on query tokens
    tokens = set(extract_terms(query))
    if not tokens:
        return []
    hits = Counter()
    for token in tokens:
        for idx in KEYWORDINDEX.get(token, []):
            hits[idx] += 1
    best_idxs = [idx for idx, _ in hits.most_common(max_items)]
    results = [CHUNKS[idx] for idx in best_idxs]
    logger.info(f"Advanced search for '{query}': {len(results)} hits")
    return results

# --- Sitemap handling ---

SITEMAP_URLS: List[str] = []
SITEMAP_SECTIONS: Dict[str, List[str]] = {}
SITEMAP_LOADED = False

def load_sitemap_local(path: str = "processed/dlh_sitemap.xml") -> Dict[str, int]:
    """
    Loads a standard XML sitemap, builds URL index/buckets.
    Returns structure: { "urls": int, "sections": int, "ok": int }
    """
    global SITEMAP_URLS, SITEMAP_SECTIONS, SITEMAP_LOADED
    from xml.etree import ElementTree as ET
    stats = {"urls": 0, "sections": 0, "ok": 0}
    try:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Sitemap not found at {path}")
            return stats
        # Assuming the provided dlh_sitemap.xml is correct and accessible
        tree = ET.parse(str(p)) 
        root = tree.getroot()
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = []
        for u in root.findall("sm:url", ns):
            loc = u.findtext("sm:loc", default="", namespaces=ns).strip()
            if loc:
                urls.append(loc)
        buckets = {}
        KEYS = [
            "impuls-workshops", "innovationsfonds", "genki", "vernetzung",
            "weiterbildung", "kuratiertes", "cops", "wb-kompass", "fobizz", "schulalltag"
        ]
        for url in urls:
            path = urllib.parse.urlparse(url).path.lower()
            for k in KEYS:
                if f"/{k}" in path:
                    buckets.setdefault(k, []).append(url)
        SITEMAP_URLS = urls
        SITEMAP_SECTIONS = buckets
        SITEMAP_LOADED = True
        stats.update(urls=len(urls), sections=len(buckets), ok=1)
        logger.info(f"Sitemap loaded with {len(urls)} URLs, {len(buckets)} buckets")
        return stats
    except Exception as e:
        logger.error(f"WARN sitemap load failed: {repr(e)}")
        return stats

# Load sitemap at startup if file exists
SITEMAP_STATS = load_sitemap_local()

def sitemap_find_innovations_tag(tag: str) -> Optional[str]:
    """Find the matching innovationsfonds tag URL from the sitemap buckets."""
    if not tag or not SITEMAP_SECTIONS:
        return None
    for url in SITEMAP_SECTIONS.get("innovationsfonds", []):
        if tag in url:
            return url
    return None

def sitemap_candidates_for_query(q: str, limit: int = 6) -> List[Dict]:
    """Returns prioritized, fake-index hits from the sitemap for relevant sections based on query. (Placeholder logic)"""
    return []

def get_ranked_with_sitemap(query: str, max_items: int = 12) -> List[Dict]:
    """Combines sitemap "boost" candidates with the core search, yielding a sorted hybrid result."""
    try:
        boosted = sitemap_candidates_for_query(query, limit=6)
    except Exception:
        boosted = []
    try:
        core = advanced_search(query, max_items=max_items)
    except Exception:
        core = []
    seen = set()
    merged = []
    def key_fn(h):
        if isinstance(h, dict):
            return h.get("url") or h.get("metadata", {}).get("source")
        if isinstance(h, tuple) and len(h) == 2 and isinstance(h[1], dict):
            hh = h[1]
            return hh.get("url") or hh.get("metadata", {}).get("source")
        return None
    for h in ensure_list(boosted) + ensure_list(core):
        u = key_fn(h)
        if not u or u in seen:
            continue
        seen.add(u)
        merged.append(h if isinstance(h, dict) else h[1])
        if len(merged) >= max_items:
            break
    return merged

# --- Live Web Scraping ------------------------------------------------------

def fetch_live_impuls_workshops() -> List[Dict]:
    """
    Liefert eine Liste von Workshops im Format:
      [{"date": datetime, "title": str, "url": str}, ...]
    Direkt von https://dlh.zh.ch/home/impuls-workshops.
    
    NOTE: Adjusted for more robust parsing based on observed structure.
    """
    events: List[Dict] = []

    try:
        r = requests.get(
            IMPULS_URL,
            timeout=15,
            headers={"User-Agent": "DLH-Bot/1.0"},
        )
        r.raise_for_status()
        html = r.text
    except Exception as ex:
        print("LIVE FETCH ERROR (Impuls: HTTP):", repr(ex))
        return []

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Offensichtlichen Müll entfernen
        for sel in ["script", "style", "noscript", ".cookie", ".consent", ".banner"]:
            for el in soup.select(sel):
                el.decompose()

        root = soup.select_one("main") or soup
        
        # Targetting the common structure: an <a> tag that contains a time and title, often in a list
        for a in root.select("li a[href], div a[href]"):
            # Try to extract elements relative to the link
            parent = a.find_parent(["li", "div"])
            
            # Extract date string
            date_str = ""
            time_el = a.find_previous_sibling("time") or parent.find("time")
            if time_el:
                date_str = time_el.get_text(" ", strip=True)
            
            # Fallback: Check if date/time info is in the surrounding text (like the provided screenshot)
            if not date_str and parent:
                # Check for sibling elements that might contain the date
                date_candidates = parent.select("span.date, div.date, time") 
                date_str = " ".join([d.get_text(" ", strip=True) for d in date_candidates])
            
            # If no date found yet, try the main link text (less reliable)
            if not date_str:
                date_str = parent.get_text(" ", strip=True)
                
            dt = parse_de_date(date_str)
            title = a.get_text(" ", strip=True) if a else ""
            href = a.get("href") if a and a.has_attr("href") else ""

            if href and href.startswith("/"):
                href = urllib.parse.urljoin(IMPULS_URL, href)

            if dt and title and href:
                events.append(
                    {
                        "date": dt,
                        "title": title,
                        "url": href or IMPULS_URL,
                    }
                )
        
        # Doppelte raus
        seen = set()
        cleaned: List[Dict] = []
        for e in events:
            # Ensure 'date' is present and hashable before using it in the key
            d_val = e.get("date")
            if not d_val:
                continue
            key = (d_val.isoformat(), e["title"])
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(e)

        print(
            f"LIVE FETCH SUCCESS (Impuls): parsed {len(cleaned)} events (raw {len(events)})"
        )
        return cleaned

    except Exception as ex:
        print("LIVE FETCH ERROR (Impuls: parse):", repr(ex))
        return []

# --- LLM Prompt Building ----------------------------------------------------

def truncates(s, n):
    """Truncate a string to n chars, safe for log/faq."""
    s = s or ""
    return s if len(s) <= n else s[:max(0, n - 1)]

def safe_snippet(snippet, max_len=MAX_SNIPPET_CHARS):
    return snippet if len(snippet) <= max_len else snippet[:max(0, max_len-1)]

def build_system_prompt() -> str:
    # Final, highly assertive system prompt for structured, detailed output
    return (
        "Du bist ein kompetenter KI-Chatbot, der Fragen rund um die DLH-Webseite dlz.zh.ch, Impuls-Workshops, Innovationsfonds-Projekte, Weiterbildungen und verwandte Bildungsthemen beantwortet. "
        "Deine Antwort **MUSS prägnant und auf Deutsch** sein. Nutze den bereitgestellten **Kontext (Chunks)** als primäre Wissensbasis. "
        "Wenn du eine Antwort aus dem Kontext generierst, **MUSST du die Quellen nennen und verlinken**. "
        "**WENN** die Frage eine Liste von Terminen, Workshops, Projekten oder Artikeln erfordert, **DANN MUSST DU** die Antwort unter Verwendung des entsprechenden HTML-Muster erzeugen, um die Links klickbar zu machen. Wenn die gesuchten Informationen fehlen, **antworte höflich, aber OHNE HTML-Struktur**.\n\n"
        "**HTML-Muster für Termine/Workshops (Timeline):**\n"
        "<section class='dlh-answer'>\n"
        "<p>Deine kurze Einleitung (1-2 Sätze, z.B. 'Hier sind die kommenden Workshops:').</p>\n"
        "<ol class='timeline'>\n"
        "<li><time>2025-11-11</time> <a href='URL' target='_blank'>Titel des Workshops</a>"
        "<div class='meta'>Ort/Format (falls bekannt)</div></li>\n"
        "\n"
        "</ol>\n"
        "<h3>Quellen</h3>\n"
        "<ul class='sources'><li><a href='URL' target='_blank'>Titel oder Domain</a></li></ul>\n"
        "</section>\n\n"
        "**HTML-Muster für Projekte/Artikel (Karten):**\n"
        "<section class='dlh-answer'>\n"
        "<p>Deine kurze Einleitung (1-2 Sätze, z.B. 'Der Innovationsfonds unterstützt folgende Projekte:').</p>\n"
        "<div class='cards'>\n"
        "<article class='card'>\n"
        "<h4><a href='URL' target='_blank'>Projekttitel</a></h4>\n"
        "<p>Kurze Beschreibung (1–2 Sätze).</p>\n"
        "</article>\n"
        "\n"
        "</div>\n"
        "<h3>Quellen</h3>\n"
        "<ul class='sources'><li><a href='URL' target='_blank'>Titel oder Domain</a></li></ul>\n"
        "</section>"
    )

def build_user_prompt(query: str, ranked: List[Dict]) -> str:
    """
    Builds user prompt including context from search results.
    Prioritizes 'content' over 'snippet' for rich context.
    """
    source_snips = []
    for ch in ranked:
        # Prioritize rich 'content' but fallback to 'snippet'
        raw_text = ch.get("content") or ch.get("snippet") or "" 
        
        # Apply truncation to keep prompt size manageable
        snippet = safe_snippet(raw_text, MAX_SNIPPET_CHARS) 
        
        url = ch.get("url", "")
        title = ch.get("title", "Quelle") 
        
        if snippet:
            # IMPORTANT: Explicitly include URL and Title in the source snippet for the LLM to use
            source_snips.append(f"Quelle: {title} (URL: {url})\n{snippet}")
            
    context = "\n---\n".join(source_snips)
    # The final prompt should clearly state the user's question and provide the context.
    return f"Frage: {query.strip()}\nKontext zur Beantwortung (verwende diese Informationen, um die Antwort zu generieren, AUCH um HTML-Links zu erstellen):\n{context}" if context else query.strip()

def build_sources(ranked: List[Dict], limit: int = 4) -> List[SourceItem]:
    """Extracts and formats sources for AnswerResponse."""
    out = []
    seen = set() 
    for ch in ranked[:limit]:
        title = ch.get("title", "(Quelle)")
        url = ch.get("url", "")
        key = (title, url)

        if key not in seen:
            # Summarization done here to get the snippet for the SourceItem model
            raw_text = ch.get('snippet') or ch.get('content', '')
            snippet = summarize_long_text(raw_text)
            out.append(
                SourceItem(
                    title=title,
                    url=url,
                    snippet=snippet
                )
            )
            seen.add(key)
    return out


def ensure_clickable_links(answer_html: str) -> str:
    """Ensure all links in LLM output are clickable (basic HTML patch, expand as needed)."""
    if not answer_html:
        return ""
    # This regex is meant to turn bare URLs into clickable links.
    return re.sub(
        r'\b(https?://[a-zA-Z0-9_\-./?=#%&]+)\b',
        r'<a href="\1" target="_blank">\1</a>',
        answer_html,
        flags=re.IGNORECASE,
    )

def validate_prompts():
    # Simple validation to check if all system/user prompts and chunk loading work as expected
    try:
        prompts = []
        for ch in CHUNKS[:2]:
            sys = build_system_prompt()
            user = build_user_prompt("Testfrage", [ch])
            prompts.append({"sys": sys, "user": user})
        logger.info("Prompt validation success")
        return {"ok": True, "prompt_count": len(prompts)}
    except Exception as e:
        logger.warning("Prompt validation failed: %s", repr(e))
        return {"ok": False, "error": str(e)}

# --- Debug and Info Endpoints -----------------------------------------------

@app.get("/debug/validate")
def debug_validate():
    """Runs prompt validation without calling OpenAI."""
    try:
        res = validate_prompts()
        logger.info(f"Validate check: {res}")
        return res
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {"ok": False, "error": str(e)}


@app.get("/debug/faq/{subject}")
def debug_faq(subject: str):
    html = build_faq_list_html(subject)
    return {"subject": subject, "html": html}

@app.get("/debug/chunks_by_tag/{tag}")
def debug_chunks_by_tag(tag: str):
    return {"tag": tag, "chunks": get_chunks_by_tag(tag)}

@app.get("/debug/sitemap")
def debug_sitemap():
    return {
        "loaded": SITEMAP_LOADED,
        "urls": len(SITEMAP_URLS),
        "sections": {k: len(v) for k, v in SITEMAP_SECTIONS.items()}
    }

@app.get("/_kb")
def kb_info():
    return {"ok": True, "chunks_loaded": CHUNKS_COUNT, "path": str(settings.chunks_path)}

@app.get("/debug/impuls")
def debug_impuls():
    ev = fetch_live_impuls_workshops()
    return {
        "count": len(ev),
        "events": [
            {
                "date": _event_to_date(e).isoformat() if _event_to_date(e) else None,
                "title": e.get("title"),
                "url": e.get("url"),
            }
            for e in ev[:10]
        ],
    }

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "DLH OpenAI API",
        "endpoints": ["/health", "/ask", "/version", "/debug/deps", "/debug/impuls", "/debug/validate", "/debug/sitemap", "/_kb"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "chunks_loaded": len(CHUNKS),
        "model": settings.openai_model
    }

@app.get("/version")
def version():
    return {
        "version": "openai-backend",
        "model": settings.openai_model,
    }

@app.get("/debug/deps")
def debug_deps():
    import platform
    import openai as _openai_pkg
    import bs4 as _bs4_pkg
    versions = {
        "python": platform.python_version(),
        "openai": getattr(_openai_pkg, "__version__", "?"),
        "bs4": getattr(_bs4_pkg, "__version__", "?"),
    }
    logger.info(f"Dependency versions: {versions}")
    return versions

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run("ultimate_api:app", host="0.0.0.0", port=8000)
    pass
