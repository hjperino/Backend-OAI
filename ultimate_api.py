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
from pydantic import BaseModel

class SourceItem(BaseModel):
    title: str
    url: str
    date: str
    description: Optional[str] = None
    date: Optional[str] = None  # or datetime if you prefer
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    content_type: Optional[str] = None
    # Add any other fields you expect

class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []

from pydantic_settings import BaseSettings

class Settings (BaseSettings):
    openai_apikey: str
    openai_model: str
    chunks_path: str
    
settings = Settings()
CHUNKS_PATH = settings.chunks_path

class SourceItem(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []

from openai import OpenAI

CHUNKSPATH = os.getenv("CHUNKSPATH", "processed/processed_chunks.json")
PROMPTCHARSBUDGET = int(os.getenv("PROMPTCHARSBUDGET", "24000"))
MAXHITSINPROMPT = 12
MAXSNIPPETCHARS = 800

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config and Initialization ---

openai_client = OpenAI(api_key=settings.openai_apikey)
app = FastAPI(title="DLH OpenAI API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://perino.info"],  # Adjust this for your deployments
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []

class QuestionRequest(BaseModel):
    question: str
    language: Optional[str] = "de"
    max_sources: Optional[int] = 3

def safe_add_lists(a, b):
    return ensure_list(a) + ensure_list(b)

def ensure_list(val):
    """Convert None to [], lists unchanged, single values/objects to [value], and AnswerResponse to list of its .sources."""
    if val is None:
        return []
    # Special handling for AnswerResponse: extract .sources if present
    if hasattr(val, "sources"):
        return ensure_list(val.sources)
    if isinstance(val, list):
        return val
    return [val]

def call_openai(system_prompt, user_prompt, max_tokens=1200):
    """Calls the OpenAI API with the specified system and user prompts. Returns the response text."""
    try:
        response = openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=max_tokens,
            stream=False,
        )
        result = response.choices[0].message.content.strip()
        logger.info(f"OpenAI returned: {repr(result)[:400]}")
        if not result.strip():
            result = "<p>Leider konnte keine Antwort generiert werden.</p>"
        return result
    except Exception as e:
        logger.error(f"OpenAI API ERROR: {repr(e)}\n{format_exc()}")
        return "<p>Fehler bei der KI-Antwort. Bitte später erneut versuchen.</p>"

def build_upcoming_workshops(chunks: List[dict]):
    today = datetime.now().date()
    upcoming = [
        ch for ch in chunks
        if ch.get('type') == 'workshop' and 'date' in ch and datetime.strptime(ch['date'], "%Y-%m-%d").date() >= today
    ]
    if not upcoming:
        return "Keine Workshops gefunden.", []
    upcoming = sorted(upcoming, key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d").date())
    answer_html = "<ul>"
    sources = []
    for event in upcoming:
        date_str = datetime.strptime(event['date'], "%Y-%m-%d").strftime("%d.%m.%Y")
        answer_html += f"<li>{date_str}: <a href='{event.get('url')}' target='_blank'>{event.get('title', 'Workshop')}</a></li>"
        sources.append(SourceItem(
            title=event.get('title', 'Workshop'),
            url=event.get('url', ''),
            snippet=event.get('text', '')
        ))
    answer_html += "</ul>"
    return answer_html, sources


def summarize_long_text(text, max_length=180):
    # If short, just return as is
    if not text or len(text) < 200:
        return text
    prompt = f"Fasse den folgenden Text in zwei Sätzen zusammen:\n{text}"
    try:
        response = openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
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
        logger.error(f"OpenAI Summarization error: {repr(e)}")
        return text[:max_length]  # Fallback: truncated text


def render_workshops_timeline_html(events, title="Kommende Impuls-Workshops"):
    """Create compact timeline HTML with clickable workshop titles."""
    if not events:
        html_str = (
            "<section class='dlh-answer'>"
            "<p>Keine Workshops gefunden.</p>"
            "<h3>Quellen</h3>"
            "<ul class='sources'>"
            "<li><a href='https://dlh.zh.ch/home/impuls-workshops' target='_blank'>Impuls-Workshop-Übersicht</a></li>"
            "</ul></section>"
        )
        logger.info(f"Returning answer: {repr(html_str)[:400]}")
        return html_str
    lis = []
    for e in events:
        d = e.get("date")
        if isinstance(d, datetime):
            d = d.date()
        date_str = d.strftime("%d.%m.%Y") if d else ""
        t = e.get("title", "Ohne Titel")
        url = e.get("url", "")
        place = e.get("place") or ""
        meta = f"<div class='meta'>{place}</div>" if place else ""
        lis.append(f"<li><time>{date_str}</time> <a href='{url}' target='_blank'>{t}</a>{meta}</li>")
    html_str = (
        "<section class='dlh-answer'>"
        f"<p>{title}:</p>"
        "<ol class='timeline'>" + "".join(lis) + "</ol>"
        "<h3>Quellen</h3>"
        "<ul class='sources'>"
        "<li><a href='https://dlh.zh.ch/home/impuls-workshops' target='_blank'>Impuls-Workshop-Übersicht</a></li>"
        "</ul></section>"
    )
    logger.info(f"Returning answer: {repr(html_str)[:400]}")
    return html_str

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    response = None
    try:
        ranked = get_ranked_with_sitemap(req.question, max_items=req.max_sources or 12)
        q_low = (req.question or "").lower().strip()

        # =============== Innovationsfonds Branch ===============
        if any(k in q_low for k in [
            "innovationsfonds", "innovations-projekt", "innovationsprojekte",
            "projektvorstellungen", "projekte", "innovationsfond projekte", "innovationsfonds projekte"
        ]):
            tag_slug = normalize_subject_to_slug(req.question)
            if tag_slug:
                tag_url = sitemap_find_innovations_tag(tag_slug)
                if tag_url:
                    cards = fetch_live_innovationsfonds_cards(tag_url)
                    if cards:
                        html_str = render_innovationsfonds_cards_html(
                            cards, subject_title=tag_slug.capitalize(), tag_url=tag_url
                        )
                        srcs = [
                            SourceItem(title=f"Innovationsfonds – {tag_slug}", url=tag_url, snippet=f"Projekte mit Tag {tag_slug}")
                        ]
                        if cards and cards[0].get("url"):
                            srcs.append(SourceItem(title=cards[0]["title"], url=cards[0]["url"], snippet=cards[0].get("snippet", "")))
                        if not html_str.strip():
                            html_str = "<p>Keine passenden Projekte gefunden.</p>"
                        logger.info(f"Returning answer: {repr(html_str)[:400]}")
                        response = AnswerResponse(answer=html_str, sources=srcs)
                        return response

        # =============== Default LLM Branch ==================
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(req.question, ranked)
        answer_html = call_openai(system_prompt, user_prompt, max_tokens=1200)
        answer_html = ensure_clickable_links(answer_html)
        sources = build_sources(ranked, limit=req.max_sources or 4)
        sources = []
        for ch in ranked[:req.max_sources or 4]:
            title = ch.get('title', 'Quelle')
            url = ch.get('url', '')
            raw_text = ch.get('snippet') or ch.get('text', '')
            snippet = summarize_long_text(raw_text)
            sources.append(SourceItem(title=title, url=url, snippet=snippet))
    except Exception as e:
        msg = "Entschuldigung, es gab einen technischen Fehler. Bitte versuchen Sie es später erneut."
        logger.error("ERROR /ask: %s\n%s", repr(e), format_exc())
        logger.info(f"Returning answer: {repr(msg)[:400]}")
        return AnswerResponse(answer=msg, sources=[])
            
    try:
        # other logic
        sources = ...  # define your sources list
        if sources and any(s.snippet for s in sources):
            answer_html = "<br><br>".join([f"<b>{s.title}</b>: {s.snippet}" for s in sources if s.snippet])
        else:
            answer_html = "<p>Leider konnte keine passende Antwort gefunden werden.</p>"

        logger.info(f"Returning answer: {repr(answer_html)[:400]}")
        response = AnswerResponse(answer=answer_html, sources=sources)
        return response

    except Exception as e:
        msg = "Entschuldigung, es gab einen technischen Fehler. Bitte versuchen Sie es später erneut."
        logger.error("ERROR /ask: %s\n%s", repr(e), format_exc())
        logger.info(f"Returning answer: {repr(msg)[:400]}")
        return AnswerResponse(answer=msg, sources=[])

# --- Load knowledge chunks (support .json and .jsonl) ---

def load_chunks(path: str) -> List[Dict]:
    """Load the knowledge base .json or .jsonl robustly."""
    out = []
    p = Path(path)
    if not p.exists():
        logger.warning(f"KB not found at {p.resolve()}")
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
            logger.error(f"Failed to load chunks from path: {e}")
            return []

# Datei laden
CHUNKS: list[dict] = load_chunks(CHUNKS_PATH)
CHUNKS_COUNT = len(CHUNKS)
logger.info(f"✅ Loaded {CHUNKS_COUNT} chunks from {CHUNKS_PATH}")

CHUNKS: List[Dict] = load_chunks(settings.chunks_path)
logger.info(f"✅ Loaded {len(CHUNKS)} chunks from {settings.chunks_path}")
PROMPT_CHARS_BUDGET = int(os.getenv("PROMPT_CHARS_BUDGET", "24000"))

# optionale Feintuning-Parameter
MAX_HITS_IN_PROMPT   = 12   # höchstens so viele Treffer einbetten
MAX_SNIPPET_CHARS    = 800  # pro Treffer; wird vor dem Einfügen gekürzt
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "processed/processed_chunks.json")

# Datei laden
CHUNKS: list[dict] = load_chunks(CHUNKS_PATH)
CHUNKS_COUNT = len(CHUNKS)
logger.info(f"✅ Loaded {CHUNKS_COUNT} chunks from {CHUNKS_PATH}")

from collections import defaultdict

def index_chunks_by_subject(chunks):
    idx = defaultdict(list)
    for i, ch in enumerate(chunks):
        subject = ch.get("subject")
        if subject:
            idx[subject.lower()].append(i)
    return idx

SUBJECTINDEX = index_chunks_by_subject(CHUNKS)

def load_chunks(path: str):
    """Lädt die Wissensbasis (.json oder .jsonl) sicher und robust."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"⚠️ KB not found at {p.resolve()}")
        return []
    if p.suffix == ".jsonl":
        out = []
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
            return response

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

def filter_chunks_by_section(chunks, section):
    return [ch for ch in chunks if ch.get("section") == section]

def get_chunks_by_tag(tag):
    return [ch for ch in CHUNKS if tag.lower() in str(ch.get("tags", [])).lower()]

def index_chunks_by_keywords(chunks):
    keywordindex = {}
    for i, ch in enumerate(chunks):
        keywords = ch.get("keywords", [])
        for kw in keywords:
            keywordindex.setdefault(kw.lower(), set()).add(i)
    return keywordindex

KEYWORDINDEX = index_chunks_by_keywords(CHUNKS)

def sort_events_chronologically(chunks, today=None):
    """Given a list of chunk dicts (with dates), split into future, past, unknown."""
    today = today or datetime.now(timezone.utc).date()
    future = []
    past = []
    nodate = []
    for ch in chunks:
        dt = None
        if "date" in ch and isinstance(ch["date"], (datetime, date)):
            dt = ch["date"].date() if isinstance(ch["date"], datetime) else ch["date"]
        elif "d" in ch and isinstance(ch["d"], (datetime, date)):
            dt = ch["d"].date() if isinstance(ch["d"], datetime) else ch["d"]
        if dt:
            if dt >= today:
                future.append((dt, ch))
            else:
                past.append((dt, ch))
        else:
            nodate.append(ch)
    past = sorted(past, key=lambda x: x[0], reverse=True)
    future = sorted(future, key=lambda x: x[0])
    return future, past, nodate

def render_event_timeline_answer(event_chunks, today=None, max_items=12):
    """Format event FAQ or date-based questions in sections: upcoming, past, and unknown."""
    future, past, nodate = sort_events_chronologically(event_chunks, today)
    future = future[:max_items]
    past = past[:max_items]
    html_parts = []
    if future:
        html_parts.append("<h3>Kommende Veranstaltungen</h3>")
        for dt, ev in future:
            html_parts.append(f"<p><strong>{dt.strftime('%d.%m.%Y')}</strong>: {ev.get('title', '(kein Titel)')}</p>")
    if past:
        html_parts.append("<h3>Vergangene Veranstaltungen</h3>")
        for dt, ev in past:
            html_parts.append(f"<p><strong>{dt.strftime('%d.%m.%Y')}</strong>: {ev.get('title', '(kein Titel)')}</p>")
    if nodate:
        html_parts.append("<h3>Weitere Informationen</h3>")
        for ev in nodate:
            html_parts.append(f"<p>{ev.get('title', '(kein Titel)')}</p>")
    answer_html = "\n".join(html_parts) if html_parts else "<p>Keine Events vorhanden.</p>"
    logger.info(f"Returning answer: {repr(answer_html)[:400]}")
    return answer_html


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

def index_chunks_by_subject(chunks):
    # Returns: subject -> [chunk ids]
    idx = defaultdict(list)
    for i, ch in enumerate(chunks):
        subject = ch.get("subject")
        if subject:
            idx[subject.lower()].append(i)
    return idx

SUBJECTINDEX = index_chunks_by_subject(CHUNKS)

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

def get_subject_faq(subject, max_items=10):
    idxs = SUBJECTINDEX.get(subject.lower(), [])
    return [CHUNKS[i] for i in idxs][:max_items]


def extractdatesfromtext(text):
    """
    Return all dates extracted from a free text (day.month.year, day.month, year, etc).
    """
    results = []
    for m in re.finditer(r'\b(\d{1,2})\.(\d{1,2})\.(20\d{2})\b', text):
        d, mth, y = m.groups()
        results.append(date(year=int(y), month=int(mth), day=int(d)))
    # Extend for more formats as needed
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

def sitemap_find_innovations_tag(tag):
    if not tag or not SITEMAP_SECTIONS:
        return None
    for url in SITEMAP_SECTIONS.get("innovationsfonds", []):
        if tag in url:
            return url
    return None

def sitemap_candidates_for_query(q: str, limit: int = 6) -> List[Dict]:
    """Returns prioritized, fake-index hits from the sitemap for relevant sections based on query."""
    if any(k in q.lower() for k in ["impuls", "workshop", "workshops"]):
        answer_html, sources = build_upcoming_workshops(CHUNKS)
        return AnswerResponse(answer=answer_html, sources=sources)

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

MONTHS_DE = dict(
    januar=1, jan=1, februar=2, feb=2, märz=3, maerz=3, mrz=3,
    april=4, apr=4, mai=5, juni=6, jun=6, juli=7, jul=7,
    august=8, aug=8, september=9, sept=9, sep=9, oktober=10,
    okt=10, november=11, nov=11, dezember=12, dez=12
)
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

def parse_de_date_text(text, refdate=None):
    """Extracts datetime from German text with fallback patterns."""
    if not text:
        return None
    t = text.strip()
    rd = refdate or datetime.now(timezone.utc)
    refyear = rd.year
    # Pattern 1: 11.11.2025
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", t)
    if m:
        d, mth, y = m.groups()
        return datetime(year=coerce_year(y, refyear), month=int(mth), day=int(d), tzinfo=timezone.utc)
    # Pattern 2: 11.11.25
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{2})", t)
    if m:
        d, mth, y = m.groups()
        return datetime(year=coerce_year(y, refyear), month=int(mth), day=int(d), tzinfo=timezone.utc)
    # Pattern 3: 25. Nov 2025 or 25. November
    m = re.match(r"(\d{1,2})\.?\s*([A-Za-zäöüÄÖÜß]+)\s*(\d{2,4})?$", t)
    if m:
        d, mon, y = m.groups()
        mth = MONTHS_DE.get(mon.lower().replace("ä","ae").replace("ö","oe").replace("ü","ue"), None)
        y = y or refyear
        if mth:
            return datetime(year=coerce_year(y, refyear), month=mth, day=int(d), tzinfo=timezone.utc)
    return None


def parse_de_date_to_date_text(date_str: str) -> Optional[date]:
    # This improves the coverage of date parsing
    try:
        # dd.mm.yyyy
        m = re.match(r"(\d{1,2})\.(\d{1,2})\.(20\d{2})", date_str)
        if m:
            d, mth, y = m.groups()
            return date(year=int(y), month=int(mth), day=int(d))
        # yyyy-mm-dd
        m = re.match(r"(20\d{2})-(\d{1,2})-(\d{1,2})", date_str)
        if m:
            y, mth, d = m.groups()
            return date(year=int(y), month=int(mth), day=int(d))
        # "12.01." with current year fallback
        m = re.match(r"(\d{1,2})\.(\d{1,2})\.", date_str)
        if m:
            d, mth = m.groups()
            return date(year=datetime.now().year, month=int(mth), day=int(d))
        # "Januar 2024"
        m = re.match(r"([A-Za-zäöüÄÖÜß]+)\s+(\d{4})", date_str)
        if m:
            mo_map = {"januar":1,"februar":2,"märz":3,"april":4,"mai":5,
                      "juni":6,"juli":7,"august":8,"september":9,
                      "oktober":10,"november":11,"dezember":12}
            mon, y = m.groups()
            if mon.lower() in mo_map:
                return date(year=int(y), month=mo_map[mon.lower()], day=1)
    except Exception:
        logger.debug(f"Could not robustly parse date string: {repr(date_str)}")
    return None

MONTHS_DE = dict(
    januar=1, jan=1, februar=2, feb=2, märz=3, maerz=3, mrz=3,
    april=4, apr=4, mai=5, juni=6, jun=6, juli=7, jul=7,
    august=8, aug=8, september=9, sept=9, sep=9, oktober=10,
    okt=10, november=11, nov=11, dezember=12, dez=12
)

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

def parse_de_date_text(text, refdate=None):
    """Extracts datetime from German text with fallback patterns."""
    if not text:
        return None
    t = text.strip()
    rd = refdate or datetime.now(timezone.utc)
    refyear = rd.year
    try:
        m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", t)
        if m:
            d, mth, y = m.groups()
            return datetime(year=coerce_year(y, refyear), month=int(mth), day=int(d), tzinfo=timezone.utc)
        m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{2})", t)
        if m:
            d, mth, y = m.groups()
            return datetime(year=coerce_year(y, refyear), month=int(mth), day=int(d), tzinfo=timezone.utc)
        m = re.match(r"(\d{1,2})\.?\s*([A-Za-zäöüÄÖÜß]+)\s*(\d{2,4})?$", t)
        if m:
            d, mon, y = m.groups()
            mth = MONTHS_DE.get(mon.lower().replace("ä","ae").replace("ö","oe").replace("ü","ue"), None)
            y = y or refyear
            if mth:
                return datetime(year=coerce_year(y, refyear), month=mth, day=int(d), tzinfo=timezone.utc)
    except Exception as ex:
        logger.debug(f"Could not robustly parse date string: {repr(text)}; error: {ex}")
    return None

def _event_to_date(e: dict) -> Optional[date]:
    """Extract date object for event sorting from dict e."""
    if not isinstance(e, dict):
        return None
    d = e.get("date") or e.get("d")
    if isinstance(d, (datetime, date)):
        return d
    date_str = e.get("when") or e.get("title")
    if date_str:
        return parse_de_date_to_date_text(date_str)
    return None

def extract_dates_from_text(text: str) -> List[Tuple[datetime, str]]:
    """Extrahiere Daten aus Text - unterstA14tzt auch abgekA14rzte Monatsnamen"""
    dates_found = []
    
    month_map_full = {
        'januar': 1, 'februar': 2, 'maerz': 3, 'april': 4,
        'mai': 5, 'juni': 6, 'juli': 7, 'august': 8,
        'september': 9, 'oktober': 10, 'november': 11, 'dezember': 12
    }
    
    month_map_abbr = {
        'jan': 1, 'feb': 2, 'mAr': 3, 'maerz': 3, 'mrz': 3, 'apr': 4,
        'mai': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'sept': 9, 'okt': 10, 'nov': 11, 'dez': 12
    }
    
    patterns = [
        (r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})', 'numeric'),
        (r'(\d{1,2})\.\s*(Januar|Februar|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s*(\d{4})', 'full_month'),
        (r'(\d{1,2})\.?\s+(Jan\.?|Feb\.?|MAr\.?|Maerz\.?|Mrz\.?|Apr\.?|Mai\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Okt\.?|Nov\.?|Dez\.?)\s+(\d{4})', 'abbr_month'),
    ]
    
    # Pattern 1: DD.MM.YYYY
    for match in re.finditer(patterns[0][0], text):
        try:
            day = int(match.group(1))
            month = int(match.group(2))
            year_str = match.group(3)
            year = int(year_str) if len(year_str) == 4 else (2000 + int(year_str))
            
            date_obj = datetime(year, month, day)
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].strip()
            
            dates_found.append((date_obj, context, match.group(0)))
        except ValueError:
            continue
    
    # Pattern 2: DD. Monat YYYY
    for match in re.finditer(patterns[1][0], text, re.IGNORECASE):
        try:
            day = int(match.group(1))
            month_name = match.group(2).lower()
            month = month_map_full.get(month_name)
            year = int(match.group(3))
            
            if month:
                date_obj = datetime(year, month, day)
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()
                
                dates_found.append((date_obj, context, match.group(0)))
        except ValueError:
            continue
    
    # Pattern 3: DD Mon. YYYY (abbreviated)
    for match in re.finditer(patterns[2][0], text, re.IGNORECASE):
        try:
            day = int(match.group(1))
            month_abbr = match.group(2).lower().replace('.', '').strip()
            month = month_map_abbr.get(month_abbr)
            year = int(match.group(3))
            
            if month:
                date_obj = datetime(year, month, day)
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()
                
                dates_found.append((date_obj, context, match.group(0)))
        except ValueError:
            continue
    
    return dates_found


def fetch_live_impuls_workshops() -> List[Dict]:
    """Fetch current and coming Impuls-Workshop events (with date normalization)."""
    IMPULS_URL = "https://dlh.zh.ch/home/impuls-workshops"
    UA = {"User-Agent": "DLH-Chatbot/1.0 https://dlh.zh.ch"}
    try:
        r = requests.get(IMPULS_URL, timeout=20, headers=UA, allow_redirects=True)
        r.raise_for_status()
        html = r.text
    except requests.RequestException as ex:
        logger.error("LIVE FETCH ERROR Impuls", exc_info=ex)
        return []

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for selector in ["script", "style", "noscript", ".cookie", ".consent", ".banner", "header", "footer", "nav", "aside"]:
        for el in soup.select(selector):
            el.decompose()

    root = soup.select_one("main") or soup
    events_raw = []
    for li in root.select("ol li, ul li"):
        text = li.get_text(" ", strip=True)
        if not text:
            continue
        a = li.find("a")
        title = a.get_text(" ", strip=True) if a else text
        href = a.get("href") if a and a.has_attr("href") else None
        if href and href.startswith("/"):
            href = requests.compat.urljoin(IMPULS_URL, href)
        time_el = li.find("time")
        when = time_el.get_text(" ", strip=True) if time_el else text
        events_raw.append(dict(title=title, url=href or IMPULS_URL, when=when))

    norm = []
    seen = set()
    for e in events_raw:
        dt = parse_de_date_to_date_text(e.get("when") or e.get("title") or "")
        if not dt:
            continue
        key = (dt.isoformat(), e["title"])
        if key in seen:
            continue
        seen.add(key)
        item = dict(e)
        item["date"] = dt
        item["d"] = dt
        norm.append(item)
    norm.sort(key=lambda x: x["d"])
    logger.info(f"LIVE FETCH SUCCESS Impuls parsed {len(norm)} events raw {len(events_raw)}")
    return norm

def get_upcoming_workshops(chunks):
    today = datetime.now().date()
    upcoming = [
        ch for ch in chunks
        if ch.get("type") == "workshop" and dateparser.parse(ch.get("date", "")) >= today
    ]
    return upcoming

def fallback_events_from_chunks() -> List[Dict]:
    """Fallback for workshops if live fetch fails: use pre-chunked events."""
    out = []
    for ch in CHUNKS:
        if ch.get("section") == "impuls-workshops" and "date" in ch and ch.get("title"):
            out.append(ch)
    logger.info(f"Loaded {len(out)} fallback events from chunks")
    return out

def normalize_subject_to_slug(q: str) -> Optional[str]:
    """Normalizes a subject/query string to a URL 'tag' slug for innovationsfonds."""
    if not q:
        return None
    q = q.lower().replace(" ", "-").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    # Simplify further normalization logic as needed
    return re.sub(r"[^a-z0-9-]", "", q.strip("-"))

def sitemap_find_innovations_tag(tag: str) -> Optional[str]:
    """Find the matching innovationsfonds tag URL from the sitemap buckets."""
    if not tag or not SITEMAP_SECTIONS:
        return None
    for url in SITEMAP_SECTIONS.get("innovationsfonds", []):
        if tag in url:
            return url
    return None

    # 2) Fallback (bekannte DLH-Struktur)
    fallback = f"https://dlh.zh.ch/home/innovationsfonds/projektvorstellungen/uebersicht/filterergebnisse-fuer-projekte/tags/{tag_slug}"
    return fallback

SUBJECT_SLUGS = {
    "chemie": "chemie",
    "physik": "physik",
    "biologie": "biologie",
    "mathematik": "mathematik",
    "informatik": "informatik",
    "deutsch": "deutsch",
    "englisch": "englisch",
    "franzoesisch": "franzoesisch",  # nur diese Schreibweise
    "italienisch": "italienisch",
    "spanisch": "spanisch",
    "geschichte": "geschichte",
    "geografie": "geografie",
    "wirtschaft": "wirtschaft",
    "recht": "recht",
    "philosophie": "philosophie",
}

def fetch_live_innovationsfonds_cards(url: str) -> List[Dict]:
    """Scrape or simulate fetching innovationsfonds project cards for a given tag/URL."""
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "lxml")
        cards = []
        for art in soup.select("article.card"):
            title = art.find("h4")
            link = art.find("a")
            snip = art.find("p")
            if not link:
                continue
            cards.append({
                "title": title.get_text(strip=True) if title else link.get_text(strip=True),
                "url": link["href"],
                "snippet": snip.get_text(strip=True) if snip else "",
            })
        logger.info(f"Fetched {len(cards)} innovationsfonds cards from live URL")
        return cards
    except Exception as e:
        logger.warning(f"Failed to fetch innovationsfonds cards: {repr(e)}")
        return []

def render_innovationsfonds_cards_html(items: List[Dict], subject_title: str, tag_url: str) -> str:
    if not items:
        html = (
            "<section class='dlh-answer'>"
            "<p>Keine Projekte gefunden.</p>"
            "<h3>Quellen</h3>"
            f"<ul class='sources'><li><a href='{tag_url}' target='_blank'>Tag-Seite</a></li></ul>"
            "</section>"
        )
        logger.info(f"Returning answer: {repr(html)[:400]}")
        return html
    cards = []
    for it in items:
        title = it.get("title", "(ohne Titel)")
        url = it.get("url", "#")
        snip = (it.get("snippet") or "").strip()
        cards.append(
            "<article class='card'>"
            f"<h4><a href='{url}' target='_blank'>{title}</a></h4>"
            f"<p>{snip}</p>"
            "</article>"
        )
    html = (
        "<section class='dlh-answer'>"
        f"<p>Innovationsfonds-Projekte im Fach <strong>{subject_title}</strong>:</p>"
        f"<div class='cards'>{''.join(cards)}</div>"
        "<h3>Quellen</h3>"
        "<ul class='sources'>"
        f"<li><a href='{tag_url}' target='_blank'>Tag-Seite: {subject_title}</a></li>"
        "</ul></section>"
    )
    logger.info(f"Returning answer: {repr(html)[:400]}")
    return html

def dedupeitems(items, key=lambda x: (x.get("title", "").lower().strip(), x.get("when", ""))):
    """Deduplicate list of event/info dicts based on title and when key."""
    seen = set()
    out = []
    for it in items:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out

def truncates(s, n):
    """Truncate a string to n chars, safe for log/faq."""
    s = s or ""
    return s if len(s) <= n else s[:max(0, n - 1)]

def safe_snippet(snippet, max_len=MAXSNIPPETCHARS):
    return snippet if len(snippet) <= max_len else snippet[:max(0, max_len-1)]


def build_system_prompt() -> str:
    return (
        
    )
def build_system_prompt() -> str:
    return (
        "Du bist ein kompetenter KI-Chatbot, der Fragen rund um die DLH-Webseite dlz.zh.ch, Impuls-Workshops, Innovationsfonds-Projekte, Weiterbildungen und verwandte Bildungsthemen beantwortet. Du gibst immer eine Antwort"
        "(…) auf. Nenne bei Terminen Datum und Zeit sowie einen verlinkten Titel. "
        "Bei Projekten im Innovationsfonds liste die Titel jeweils als klickbare Links. "
        "Wenn es sich um Termine/Workshops handelt, nutze folgendes HTML-Muster:\n\n"
        "Kurz-Einleitung (1 Satz).\n"
        "<ol class='timeline'>\n"
        "<li><time>2025-11-11</time> <a href='URL' target='_blank'>Titel des Workshops</a>"
        "<div class='meta'>Ort/Format (falls bekannt)</div></li>\n"
        "</ol>\n"
        "<h3>Quellen</h3>\n"
        "<ul class='sources'><li><a href='URL' target='_blank'>Titel oder Domain</a></li></ul>\n"
        "\n"
        "Bei Projektlisten nutze Karten:\n\n"
        "<div class='cards'>\n"
        "<article class='card'>\n"
        "<h4><a href='URL' target='_blank'>Projekttitel</a></h4>\n"
        "<p>Kurze Beschreibung (1–2 Sätze).</p>\n"
        "</article>\n"
        "</div>\n"
        "<h3>Quellen</h3>\n"
        "<ul class='sources'><li><a href='URL' target='_blank'>Titel oder Domain</a></li></ul>\n"
    )
    
def build_user_prompt(query: str, ranked: List[Dict]) -> str:
    """Builds user prompt including context from search results."""
    source_snips = []
    for ch in ranked:
        snippet = ch.get("snippet", "")
        url = ch.get("url", "")
        if snippet:
            source_snips.append(f"Quelle: {url}\n{snippet}")
    context = "\n".join(source_snips)
    return f"{query.strip()}\nKontext:\n{context}" if context else query.strip()

def build_sources(ranked: List[Dict], limit: int = 4) -> List[SourceItem]:
    """Extracts and formats sources for AnswerResponse."""
    out = []
    for ch in ranked[:limit]:
        out.append(
            SourceItem(
                title=ch.get("title", "(Quelle)"),
                url=ch.get("url", ""),
                snippet=ch.get("snippet", "")
            )
        )
    return out

def create_enhanced_prompt(question: str, chunks: List[Dict], intent: Dict) -> str:
    """Erstelle Prompt - Formatierung ist im System Prompt"""
    
    current_date = datetime.now()
    current_date_str = current_date.strftime('%d.%m.%Y')
    
    # Event-Sortierung von gestern!
    if intent['is_date_query'] or any(keyword in ['workshop', 'veranstaltung'] for keyword in intent['topic_keywords']):
        sorted_events = sort_events_chronologically(chunks, current_date)
        
        context_parts = []
        
        if sorted_events['future_events']:
            context_parts.append("=== KOMMENDE VERANSTALTUNGEN (chronologisch sortiert) ===")
            for event in sorted_events['future_events']:
                days_until = (event['date'].date() - current_date.date()).days
                context_parts.append(f"\nY... DATUM: {event['date'].strftime('%d.%m.%Y (%A)')} (in {days_until} Tagen)")
                context_parts.append(f"Titel: {event['chunk']['metadata'].get('title', 'Unbekannt')}")
                context_parts.append(f"Quelle: {event['chunk']['metadata'].get('source', 'Unbekannt')}")
                context_parts.append(event['chunk']['content'][:400])
                context_parts.append("---")
        
        if sorted_events['past_events']:
            context_parts.append("\n\n=== VERGANGENE VERANSTALTUNGEN ===")
            for event in sorted_events['past_events'][:5]:
                days_ago = (current_date.date() - event['date'].date()).days
                context_parts.append(f"\nY... DATUM: {event['date'].strftime('%d.%m.%Y (%A)')} (vor {days_ago} Tagen - BEREITS VORBEI)")
                context_parts.append(f"Titel: {event['chunk']['metadata'].get('title', 'Unbekannt')}")
                context_parts.append(f"Quelle: {event['chunk']['metadata'].get('source', 'Unbekannt')}")
                context_parts.append(event['chunk']['content'][:400])
                context_parts.append("---")
        
        if sorted_events['no_date_events']:
            context_parts.append("\n\n=== WEITERE INFORMATIONEN ===")
            for item in sorted_events['no_date_events']:
                context_parts.append(f"\nTitel: {item['chunk']['metadata'].get('title', 'Unbekannt')}")
                context_parts.append(f"Quelle: {item['chunk']['metadata'].get('source', 'Unbekannt')}")
                context_parts.append(item['chunk']['content'][:400])
                context_parts.append("---")
        
        context = "\n".join(context_parts)
        
        prompt = f"""
Heutiges Datum: {current_date_str}
Bitte beantworte die folgende Frage mit Bezug auf die gegebenen Daten.
{question}
"""
        return prompt


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

@app.get("/debug/impuls")
def debug_impuls():
    ev = fetch_live_impuls_workshops()
    logger.info(f"Debug impuls: {len(ev)} events")
    return {"count": len(ev), "sample": ev[:3]}
    
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
    import lxml as _lxml_pkg
    versions = {
        "python": platform.python_version(),
        "openai": getattr(_openai_pkg, "__version__", "?"),
        "bs4": getattr(_bs4_pkg, "__version__", "?"),
        "lxml": getattr(_lxml_pkg, "__version__", "?"),
    }
    logger.info(f"Dependency versions: {versions}")
    return versions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ultimate_api:app", host="0.0.0.0", port=8000)
