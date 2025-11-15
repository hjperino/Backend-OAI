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
    """
    openai_apikey: str
    openai_model: str 
    chunks_path: str = "processed/processed_chunks.json" 
    structured_db_path: str = "processed/structured_db.json" # NEUER PFAD

settings = Settings()
CHUNKS_PATH = settings.chunks_path
STRUCTURED_DB_PATH = settings.structured_db_path # NEU

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
# (Wird für Live-Scraping und DB-ISO-Strings benötigt)

DMY_DOTTED_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b")
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

# --- NEU: SUBJECT MAP (Für Innovationsfonds-Filterung) ---
SUBJECT_MAP = {
    'abu': 'ABU', 'architektur': 'Architektur EFZ', 'automobilberufe': 'Automobilberufe',
    'berufskunde': 'Berufskunde', 'bildnerisches-gestalten': 'Bildnerisches Gestalten',
    'biologie': 'Biologie', 'brueckenangebot': 'Bruckenangebot', 'chemie': 'Chemie',
    'coiffeuse': 'Coiffeuse-Coiffeur', 'deutsch': 'Deutsch', 'eba': 'EBA',
    'elektroberufe': 'Elektroberufe', 'englisch': 'Englisch', 'fage': 'FaGe',
    'franzoesisch': 'Franzosisch', 'geographie': 'Geographie',
    'geomatiker': 'Geomatiker:innen EFZ', 'geschichte': 'Geschichte',
    'geschichte-und-politik': 'Geschichte und Politik', 'griechisch': 'Griechisch',
    'ika': 'IKA', 'informatik': 'Informatik', 'italienisch': 'Italienisch',
    'landwirtschaftsmechaniker': 'Landwirtschaftsmechaniker:innen', 'latein': 'Latein',
    'mathematik': 'Mathematik', 'maurer': 'Maurer:innen', 'physik': 'Physik',
    'russisch': 'Russisch', 'schreiner': 'Schreiner:in',
    'sozialwissenschaften': 'Sozialwissenschaften', 'spanisch': 'Spanisch',
    'sport': 'Sport', 'ueberfachlich': 'Uberfachlich', 'wirtschaft': 'Wirtschaft'
}
# Erstelle eine normalisierte Map für die Suche (z.B. 'chemie' -> 'chemie')
NORMALIZED_SUBJECT_MAP = {
    re.sub(r'[^a-z]', '', v.lower().replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')): k 
    for k, v in SUBJECT_MAP.items()
}
# Füge die Schlüssel hinzu (z.B. 'abu' -> 'abu')
for k in SUBJECT_MAP.keys():
    NORMALIZED_SUBJECT_MAP[k] = k

# --- NEU: HARDCODED CoP-Datenbank ---
COPS_URL = "https://dlh.zh.ch/home/cops"
COPS_HARDCODED_DB = [
    {"title": "Allgemeinbildender Unterricht", "description": "Leitung: Erika Langhans", "url": COPS_URL},
    {"title": "Chemie", "description": "Leitung: Amadeus Bärtsch", "url": COPS_URL},
    {"title": "Deutsch", "description": "Leitung: Carmen Aus der Au, Natalija Jovanovic", "url": COPS_URL},
    {"title": "Englisch", "description": "Leitung: Franziska Tobler, Marija Josifovic", "url": COPS_URL},
    {"title": "Entwicklungsteamleitungen", "description": "Leitung: Hansjürg Perino", "url": COPS_URL},
    {"title": "Fachkundige indiv. Begleitung FiB", "description": "Leitung: Nadine Vetterli", "url": COPS_URL},
    {"title": "Gamification", "description": "Leitung: Benaja Schellenberg", "url": COPS_URL},
    {"title": "GenKI", "description": "Leitung: Pascal Schmidt", "url": COPS_URL},
    {"title": "Geografie", "description": "Leitung: Patrik Weiss", "url": COPS_URL},
    {"title": "Geschichte", "description": "Leitung: Justine Burkhalter", "url": COPS_URL},
    {"title": "Informatik", "description": "Leitung: Theresa Luternauer", "url": COPS_URL},
    {"title": "KV", "description": "Leitung: Anita Schuler", "url": COPS_URL},
    {"title": "Lehrpersonen Prävention und Gesundheitsförderung", "description": "Leitung: Christoph Staub", "url": COPS_URL},
    {"title": "Medien- und Informationskompetenz", "description": "Leitung: Monica Bronner", "url": COPS_URL},
    {"title": "Meta", "description": "Leitung: Christof Glaus", "url": COPS_URL},
    {"title": "Moodle Admin", "description": "Leitung: Thomas Korner", "url": COPS_URL},
    {"title": "Moodle Lehrpersonen", "description": "Leitung: Simon Küpfer", "url": COPS_URL},
    {"title": "Religionen, Kulturen, Ethik", "description": "Leitung: Christoph Staub", "url": COPS_URL},
    {"title": "Romanistik", "description": "Leitung: Francisca Ruiz (Spanisch), Letizia Martini (Italienisch), Ariane Chaoui Nowik (Französisch)", "url": COPS_URL}
]


# --- Global Constants & Initialization --------------------------------------

from openai import OpenAI
IMPULS_URL = "https://dlh.zh.ch/home/impuls-workshops" 

openai_client = OpenAI(api_key=settings.openai_apikey)
app = FastAPI(title="DLH OpenAI API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://perino.info"], # Passen Sie dies für Ihre Domains an
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
    """
    Zusammenfassen von langem Text. Behebt das 'max_tokens' vs 'max_completion_tokens' Problem.
    """
    # If short, just return as is
    if not text or len(text) < 200:
        # Wenn der Text kurz ist, nur Zeilenumbrüche entfernen
        return " ".join(text.splitlines()) 
        
    prompt = f"Fasse den folgenden Text in zwei Sätzen zusammen:\n{text}"
    try:
        # 1. VERSUCH: Verwende max_completion_tokens (für 'gpt-5')
        response = openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_length,
            stream=False
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        error_msg = str(e)
        # 2. FALLBACK: Wenn 'max_completion_tokens' fehlschlägt, versuche 'max_tokens'
        if 'unsupported parameter' in error_msg.lower() and 'max_tokens' in error_msg.lower():
            logger.warning(f"Summarize: Model {settings.openai_model} rejected max_completion_tokens. Falling back to max_tokens.")
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
            except Exception as e_fallback:
                logger.error(f"OpenAI Summarization error (Fallback failed): {repr(e_fallback)}")
                return text[:max_length] + "..." # Fallback: Text kürzen
        
        # Anderer Fehler
        logger.error(f"OpenAI Summarization error: {repr(e)}")
        return text[:max_length] + "..." # Fallback: Text kürzen

def _event_to_date(e: Dict) -> Optional[datetime]:
    """
    Hilfsfunktion für die Workshop-Intentlogik:
    Nimmt ein Event-Dict (aus der DB oder Live) und gibt ein datetime-Objekt zurück.
    """
    if not isinstance(e, dict):
        return None

    # Priorität 1: Bereits geparstes Objekt
    d_obj = e.get("_d")
    if isinstance(d_obj, datetime):
        return d_obj

    # Priorität 2: ISO-String aus der DB
    d_iso = e.get("date_iso")
    if d_iso:
        try:
            return datetime.fromisoformat(d_iso)
        except ValueError:
            pass

    # Priorität 3: Live-Scraping-Text (Fallback)
    txt = e.get("when") or e.get("title") or ""
    return parse_de_date(txt)

# --- Date Parsing Helpers ---------------------------------------------------
# (Diese werden nur noch vom Live-Scraper benötigt)

def coerce_year(y, refyear):
    if not y: return refyear
    y = y.strip()
    if len(y) == 2: y = "20" + y
    try: return int(y)
    except Exception: return refyear

def parse_de_date(text: str, ref_date: Optional[datetime] = None) -> Optional[datetime]:
    if not text: return None
    t = text.strip()
    rd = ref_date or datetime.now(timezone.utc)
    ref_year = rd.year

    m = DMY_DOTTED_RE.search(t)
    hh, mm = None, None
    tm = TIME_RE.search(t)
    if tm: hh, mm = int(tm.group(1)), int(tm.group(2))
    if m:
        d = int(m.group(1)); mth = int(m.group(2)); y = coerce_year(m.group(3), ref_year)
        try:
            base = datetime(y, mth, d, tzinfo=timezone.utc)
            if hh is not None: base = base.replace(hour=hh, minute=mm or 0)
            return base
        except Exception: pass

    m = DMY_TEXT_RE.search(t)
    if m:
        d = int(m.group(1))
        month_word = (m.group(2) or "").strip().lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        mth = MONTHS_DE.get(month_word)
        y = coerce_year(m.group(3), ref_year)
        if mth:
            try:
                base = datetime(y, mth, d, tzinfo=timezone.utc)
                if hh is not None: base = base.replace(hour=hh, minute=mm or 0)
                return base
            except Exception: pass
    return None

# --- HTML Renderers (Deterministisch) ---------------------------------------

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
        
        # KORREKTUR: Bereinige den Titel von Datums-Artefakten aus 'build_database.py'
        t = e.get("title", "Ohne Titel")
        # Entfernt (z.B. " (20. Nov 2025)" oder " (20.11.2025)")
        t = re.sub(r'\s*\(\s*\d{1,2}\.\s*[A-Za-zäöüÄÖÜ]+\s*\d{0,4}\s*\)$', '', t).strip() 
        t = re.sub(r'\s*\(\s*\d{1,2}\.\d{1,2}\.\d{2,4}\s*\)$', '', t).strip()
        
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
        
        # KORREKTUR (DESIGN): Füge CSS für "Box"-Darstellung hinzu (Abstand/Rahmen)
        cards_html.append(
            f"<article class='card' style='margin-bottom: 15px; border: 1px solid #eee; padding: 10px; border-radius: 5px;'>"
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

# --- Knowledge Base Loading (Structured and Unstructured) -------------------

def load_chunks(path: str) -> List[Dict]:
    """Load the (unstructured) knowledge base .json or .jsonl robustly."""
    out = []
    p = Path(path)
    if not p.exists():
        logger.warning(f"⚠️ Unstrukturierte KB nicht gefunden: {p.resolve()}")
        return []
    
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: out.append(json.loads(line))
                except Exception: logger.debug(f"Skipping invalid JSON line in {path}")
        return out
    else:
        try:
            return json.load(p.open("r", encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load chunks from {path}: {e}")
            return []

def load_structured_db(path: str) -> Dict:
    """Lädt die neue, vorverarbeitete strukturierte Datenbank."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"⚠️ Strukturierte DB NICHT GEFUNDEN: {path}. Antworten für Listen werden fehlschlagen.")
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Wichtig: Datums-Strings in echten datetime-Objekte umwandeln
            if "workshops_kb" in data:
                for event in data["workshops_kb"]:
                    try:
                        # Stellt sicher, dass das _d-Feld für die Sortierung existiert
                        event["_d"] = datetime.fromisoformat(event["date_iso"])
                    except (ValueError, TypeError):
                        event["_d"] = None # Fallback
            logger.info(f"✅ Strukturierte DB geladen ({path})")
            return data
    except Exception as e:
        logger.error(f"Fehler beim Laden der strukturierten DB: {e}")
        return {}

# Load chunks (für RAG-Fallback)
CHUNKS: List[Dict] = load_chunks(CHUNKS_PATH)
CHUNKS_COUNT = len(CHUNKS)
logger.info(f"✅ {CHUNKS_COUNT} Chunks (für RAG) geladen von {CHUNKS_PATH}")

# Load structured DB (für deterministische Antworten)
STRUCTURED_DB: Dict = load_structured_db(STRUCTURED_DB_PATH)


# --- RAG Fallback Functions (Keyword Search & Prompting) ----------------

def index_chunks_by_keywords(chunks):
    keywordindex = {}
    for i, ch in enumerate(chunks):
        keywords = ch.get("keywords", [])
        for kw in keywords:
            keywordindex.setdefault(kw.lower(), set()).add(i)
    return keywordindex

KEYWORDINDEX = index_chunks_by_keywords(CHUNKS)

def extract_terms(query: str) -> Set[str]:
    """Simple term extraction for keyword indexing."""
    query = query.lower()
    tokens = re.split(r"[^a-zäöüß0-9]+", query)
    return {t for t in tokens if len(t) > 2 and t not in ["der", "die", "das", "und", "oder"]}

def advanced_search(query, max_items=12):
    # Score chunks based on query tokens
    tokens = set(extract_terms(query))
    if not tokens:
        return []
    hits = Counter()
    for token in tokens:
        for idx in KEYWORDINDEX.get(token, []):
            if idx < len(CHUNKS): # Sicherstellen, dass der Index gültig ist
                hits[idx] += 1
    best_idxs = [idx for idx, _ in hits.most_common(max_items)]
    results = [CHUNKS[idx] for idx in best_idxs if idx < len(CHUNKS)]
    logger.info(f"Advanced search (RAG) for '{query}': {len(results)} hits")
    return results

def get_ranked_with_sitemap(query: str, max_items: int = 12) -> List[Dict]:
    """Wrapper für die advanced_search, da Sitemap-Logik nicht verwendet wird."""
    return advanced_search(query, max_items=max_items)

# --- Sitemap handling (Wird geladen, aber nicht aktiv für die Suche genutzt) ---

SITEMAP_URLS: List[str] = []
SITEMAP_SECTIONS: Dict[str, List[str]] = {}
SITEMAP_LOADED = False

def load_sitemap_local(path: str = "processed/dlh_sitemap.xml") -> Dict[str, int]:
    """
    Loads a standard XML sitemap, builds URL index/buckets.
    """
    global SITEMAP_URLS, SITEMAP_SECTIONS, SITEMAP_LOADED
    from xml.etree import ElementTree as ET
    stats = {"urls": 0, "sections": 0, "ok": 0}
    try:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Sitemap not found at {path}")
            return stats
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

SITEMAP_STATS = load_sitemap_local()

# --- Live Web Scraping (Nur als Fallback für Workshops) --------------------

def fetch_live_impuls_workshops() -> List[Dict]:
    """
    Liefert eine Liste von Workshops durch Live-Scraping.
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
        for sel in ["script", "style", "noscript", ".cookie", ".consent", ".banner"]:
            for el in soup.select(sel):
                el.decompose()
        root = soup.select_one("main") or soup
        
        for a in root.select("li a[href], div a[href]"):
            parent = a.find_parent(["li", "div"])
            date_str = ""
            time_el = a.find_previous_sibling("time") or (parent and parent.find("time"))
            if time_el:
                date_str = time_el.get_text(" ", strip=True)
            
            if not date_str and parent:
                date_candidates = parent.select("span.date, div.date, time") 
                date_str = " ".join([d.get_text(" ", strip=True) for d in date_candidates])
            
            if not date_str:
                date_str = parent.get_text(" ", strip=True)
                
            dt = parse_de_date(date_str)
            title = a.get_text(" ", strip=True) if a else ""
            href = a.get("href") if a and a.has_attr("href") else ""

            if href and href.startswith("/"):
                href = urllib.parse.urljoin(IMPULS_URL, href)

            if dt and title and href:
                events.append({"date": dt, "title": title, "url": href or IMPULS_URL})
        
        seen = set()
        cleaned: List[Dict] = []
        for e in events:
            d_val = e.get("date")
            if not d_val: continue
            key = (d_val.isoformat(), e["title"])
            if key in seen: continue
            seen.add(key)
            cleaned.append(e)

        print(f"LIVE FETCH SUCCESS (Impuls): parsed {len(cleaned)} events (raw {len(events)})")
        return cleaned

    except Exception as ex:
        print("LIVE FETCH ERROR (Impuls: parse):", repr(ex))
        return []

# --- LLM Prompt Building (Nur für RAG-Fallback) -----------------------------

def build_system_prompt() -> str:
    # (Behält den strengen Prompt für den Fallback bei)
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
        raw_text = ch.get("content") or ch.get("snippet") or "" 
        snippet = safe_snippet(raw_text, MAX_SNIPPET_CHARS) 
        url = ch.get("url", "")
        title = ch.get("title", "Quelle") 
        if snippet:
            source_snips.append(f"Quelle: {title} (URL: {url})\n{snippet}")
            
    context = "\n---\n".join(source_snips)
    return f"Frage: {query.strip()}\nKontext zur Beantwortung (verwende diese Informationen, um die Antwort zu generieren, AUCH um HTML-Links zu erstellen):\n{context}" if context else query.strip()

def build_sources(ranked: List[Dict], limit: int = 4):
    """Extracts and formats sources for AnswerResponse."""
    out = []
    seen = set() 
    for ch in ranked[:limit]:
        title = ch.get("title", "(Quelle)")
        url = ch.get("url", "")
        key = (title, url)
        if key not in seen:
            raw_text = ch.get('snippet') or ch.get('content', '')
            snippet = summarize_long_text(raw_text)
            out.append(SourceItem(title=title, url=url, snippet=snippet))
            seen.add(key)
    return out


def ensure_clickable_links(answer_html: str) -> str:
    """Ensure all links in LLM output are clickable (basic HTML patch, expand as needed)."""
    if not answer_html:
        return ""
    return re.sub(
        r'\b(https?://[a-zA-Z0-9_\-./?=#%&]+)\b',
        r'<a href="\1" target="_blank">\1</a>',
        answer_html,
        flags=re.IGNORECASE,
    )

# --- Main API Endpoint (Neue Logik) ------------------------------------------------------

def _normalize_query_for_subjects(q: str) -> List[str]:
    """Wandelt eine Anfrage wie 'Wirtschaft und Recht' in ['wirtschaft', 'recht'] um."""
    q = q.lower().replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
    q_tokens = re.split(r"[^a-z]+", q)
    
    found_subjects = []
    for token in q_tokens:
        if token in NORMALIZED_SUBJECT_MAP:
            found_subjects.append(NORMALIZED_SUBJECT_MAP[token]) # z.B. 'chemie'
            
    return found_subjects

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    try:
        q_low = (req.question or "").lower()
        
        # ---- 1. DETERMINISTISCHER HANDLER (NEU) ----
        # (Verwendet die geladene STRUCTURED_DB)

        if not isinstance(STRUCTURED_DB, dict):
            logger.error("STRUCTURED_DB ist keine Dict oder nicht geladen. Ladefehler? Fallback zu RAG.")
            return rag_fallback(req, q_low) # Springe zum RAG-Fallback

        # 1a. Innovationsfonds Projects (Direct Python Rendering)
        if any(k in q_low for k in ["innovationsfonds", "projekte", "innovation"]):
            
            all_projects = STRUCTURED_DB.get("innovationsfonds_projects", [])
            projects_to_show = all_projects
            title = "Alle DLH Innovationsfonds Projekte"
            
            # KORREKTUR: Filterung nach Fach
            subjects_in_query_keys = _normalize_query_for_subjects(q_low)
            
            if subjects_in_query_keys:
                logger.info(f"Filtere Innovationsfonds nach Fächern: {subjects_in_query_keys}")
                filtered_projects = []
                for proj in all_projects:
                    # Prüfe, ob eines der gesuchten Fächer in den Fächern des Projekts enthalten ist
                    if any(subj_key in proj.get('subjects', []) for subj_key in subjects_in_query_keys):
                        filtered_projects.append(proj)
                
                if filtered_projects:
                    projects_to_show = filtered_projects
                    # Erstelle einen Titel basierend auf den gefundenen Fächern
                    subject_names = [SUBJECT_MAP.get(key, key) for key in subjects_in_query_keys]
                    title = f"Innovationsfonds Projekte (Filter: {', '.join(subject_names)})"
                else:
                    # Wenn Filter gesetzt, aber nichts gefunden wurde
                    projects_to_show = [] 
                    subject_names = [SUBJECT_MAP.get(key, key) for key in subjects_in_query_keys]
                    title = f"Innovationsfonds Projekte (Keine Treffer für: {', '.join(subject_names)})"
            
            # Rendere die (gefilterte oder ungefilterte) Liste
            html_answer = render_structured_cards_html(
                projects_to_show, 
                title=title, 
                source_url="https://dlh.zh.ch/home/innovationsfonds/projektvorstellungen/uebersicht"
            )
            returned_sources = [SourceItem(title=p['title'], url=p['url']) for p in projects_to_show[:req.max_sources or 4]]
            return AnswerResponse(
                answer=html_answer,
                sources=returned_sources
            )
        
        # 1b. Fobizz Resources (Direct Python Rendering)
        if any(k in q_low for k in ["fobizz", "tool", "sprechstunde", "kurs", "weiterbildung"]):
            fobizz_items = STRUCTURED_DB.get("fobizz_resources", [])
            if fobizz_items:
                html_answer = render_structured_cards_html(
                    fobizz_items, 
                    title="DLH Fobizz Angebote", 
                    source_url="https://dlh.zh.ch/home/wb-kompass/wb-angebote/1007-wb-plattformen"
                )
                returned_sources = [SourceItem(title=p['title'], url=p['url']) for p in fobizz_items[:req.max_sources or 4]]
                return AnswerResponse(
                    answer=html_answer,
                    sources=returned_sources
                )
        
        # 1c. CoPs (Communities of Practice) (NEU)
        if any(k in q_low for k in ["cops", "communities of practice", "community"]):
            cops_items = COPS_HARDCODED_DB # Greife auf die hardcodierte Liste zu
            if cops_items:
                html_answer = render_structured_cards_html(
                    cops_items, 
                    title="DLH Communities of Practice (CoPs)", 
                    source_url="https://dlh.zh.ch/home/cops"
                )
                returned_sources = [SourceItem(title=p['title'], url=p['url']) for p in cops_items[:req.max_sources or 4]]
                return AnswerResponse(
                    answer=html_answer,
                    sources=returned_sources
                )

        # 1d. WORKSHOP INTENT (Time-sensitive, aus DB)
        if any(k in q_low for k in ["impuls", "workshop", "workshops", "termine"]):
            
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
            
            # --- CHUNK ANALYSIS (aus STRUCTURED_DB) ---
            all_events = STRUCTURED_DB.get("workshops_kb", [])
            
            if all_events:
                today = datetime.now(timezone.utc).date()
                # Filtern der Events (benötigt _d, das beim Laden der DB erstellt wurde)
                future_chunk_events = [e for e in all_events if e.get("_d") and e["_d"].date() >= today]
                past_chunk_events = [e for e in all_events if e.get("_d") and e["_d"].date() < today]

                print(f"Y Chunks found: future={len(future_chunk_events)}, past={len(past_chunk_events)}")
                
                if want_next:
                    events_to_show = sorted(future_chunk_events, key=lambda x: x["_d"])[:1]
                    title_suffix = " (aus KB)"
                elif want_past:
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

                return AnswerResponse(
                    answer=html,
                    sources=[SourceItem(title=e['title'], url=e['url']) for e in events_to_show if 'title' in e and 'url' in e] 
                )

            # 4. Fallback to LIVE SCRAPER if DB is empty
            
            events = fetch_live_impuls_workshops()
            norm_events = []
            for e in events:
                d = _event_to_date(e)
                if d: ee = dict(e); ee["_d"] = d; norm_events.append(ee)

            today = datetime.now(timezone.utc).date()
            future = [e for e in norm_events if e["_d"].date() >= today]
            past = [e for e in norm_events if e["_d"].date() < today]

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
            
            return AnswerResponse(answer=html, sources=[SourceItem(title="Impuls-Workshop-Übersicht", url=IMPULS_URL)])
            

        # ---- 3. DEFAULT LLM/RAG BRANCH (General Concepts, Definitions, etc.) ----
        return rag_fallback(req, q_low)

    except Exception as e:
        msg = "Entschuldigung, es gab einen technischen Fehler. Bitte versuchen Sie es später erneut."
        logger.error(f"ERROR /ask: %s\n{format_exc()}", repr(e))
        logger.info(f"Returning answer: {repr(msg)[:400]}")
        return AnswerResponse(answer=msg, sources=[])

def rag_fallback(req: QuestionRequest, q_low: str):
    """
    Diese Funktion wird aufgerufen, wenn keine der deterministischen Regeln zutrifft.
    Sie verwendet die RAG-Pipeline (Chunks + LLM), um eine Antwort auf Freitextfragen zu generieren.
    """
    logger.info(f"Kein strukturierter Handler für '{q_low}'. Fallback zu RAG/LLM.")
    
    # Führe die RAG-Suche (advanced_search) nur aus, wenn sie benötigt wird
    ranked = get_ranked_with_sitemap(req.question, max_items=req.max_sources or 4)
    
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

# --- Debug and Info Endpoints (Restored from original file) ---

def validate_prompts():
    try:
        prompts = []
        for ch in CHUNKS[:2]: # Verwendet die RAG-Chunks
            sys = build_system_prompt()
            user = build_user_prompt("Testfrage", [ch])
            prompts.append({"sys": sys, "user": user})
        logger.info("Prompt validation success")
        return {"ok": True, "prompt_count": len(prompts)}
    except Exception as e:
        logger.warning("Prompt validation failed: %s", repr(e))
        return {"ok": False, "error": str(e)}

@app.get("/debug/validate")
def debug_validate_endpoint():
    return validate_prompts()

# (Definiere SUBJECTINDEX global, damit es in der FAQ-Funktion verwendet werden kann)
def index_chunks_by_subject(chunks):
    idx = defaultdict(list)
    for i, ch in enumerate(chunks):
        subject = ch.get("subject")
        if subject:
            idx[subject.lower()].append(i)
    return idx
SUBJECTINDEX = index_chunks_by_subject(CHUNKS)


@app.get("/debug/faq/{subject}")
def debug_faq(subject: str):
    
    def get_subject_faq(subject, max_items=10):
        idxs = SUBJECTINDEX.get(subject.lower(), [])
        return [CHUNKS[i] for i in idxs if i < len(CHUNKS)] # Sicherstellen, dass der Index gültig ist

    faqs = get_subject_faq(subject)
    arts = []
    for faq in faqs:
        title = faq.get("title", "(keine Überschrift)")
        url = faq.get("url", "#")
        snippet = faq.get("snippet", "")
        arts.append(f"<article><h4><a href='{url}' target='_blank'>{title}</a></h4><p>{snippet}</p></article>")
        
        max_length = 1200 # Standardwert definieren
        if sum(len(a) for a in arts) > max_length:
            break
            
    html = "<div class='faq-list'>" + "\n".join(filter(None, ensure_list(arts))) + "</div>" if arts else "<p>Keine FAQs gefunden.</p>"
    return {"subject": subject, "html": html}

@app.get("/debug/chunks_by_tag/{tag}")
def debug_chunks_by_tag(tag: str):
    def get_chunks_by_tag(tag):
        tag_lower = tag.lower()
        return [ch for ch in CHUNKS if tag_lower in str(ch.get("tags", [])).lower()]
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
    db_status = "OK"
    if not isinstance(STRUCTURED_DB, dict) or not STRUCTURED_DB:
        db_status = "FEHLER: structured_db.json nicht gefunden oder leer."

    return {
        "ok": True, 
        "chunks_loaded (RAG)": CHUNKS_COUNT, 
        "path": str(settings.chunks_path),
        "structured_db_status": db_status
    }

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
    # Prüft auch, ob die strukturierte DB geladen wurde
    db_loaded = bool(STRUCTURED_DB)
    db_status = "OK"
    if not db_loaded:
        db_status = "FEHLER: structured_db.json nicht gefunden oder leer."
    
    return {
        "status": "healthy",
        "chunks_loaded (RAG)": CHUNKS_COUNT,
        "structured_db_status": db_status,
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
