# DLH Chatbot API (OpenAI) — full file
import os
import json
import re
import urllib.parse
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from traceback import format_exc

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    try:
        ranked = advanced_search(req.question, max_items=req.max_sources or 8)
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(req.question, ranked)

        print("Y  LLM call → model:", OPENAI_MODEL, "| prompt_len:", len(user_prompt))
        answer = call_openai(system_prompt, user_prompt, max_tokens=1200)
        print("Y  LLM OK")

        sources = build_sources(ranked, limit=req.max_sources or 3)
        return AnswerResponse(answer=answer, sources=sources)

    except Exception as e:
        # Vollständiges Traceback ins Render-Log:
        print("ERROR in /ask:", repr(e))
        print(format_exc())

        # Nutzerfreundlicher Fallback – immer schema-konform:
        msg = "<strong>Entschuldigung, es gab einen technischen Fehler.</strong><br>Bitte versuchen Sie es in Kürze erneut."
        return AnswerResponse(answer=msg, sources=[])
# -----------------------------
# OpenAI client
# -----------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID") or None,
)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="DLH Chatbot API (OpenAI)")

# CORS for browser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://perino.info","https://www.perino.info"],
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class QuestionRequest(BaseModel):
    question: str
    language: Optional[str] = "de"
    max_sources: Optional[int] = 8

class Source(BaseModel):
    url: str
    title: str
    snippet: str = ""

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source] = []

# -----------------------------
# Load processed chunks
# -----------------------------
CHUNKS: List[Dict] = []
ROOT = Path(__file__).parent
for candidate in [
    ROOT / "processed" / "processed_chunks.json",
    ROOT / "processed_chunks.json",
    Path("/mnt/data/processed_chunks.json"),
]:
    if candidate.exists():
        try:
            CHUNKS = json.loads(candidate.read_text(encoding="utf-8"))
            print(f"Loaded {len(CHUNKS)} chunks from {candidate}")
            break
        except Exception as e:
            print("Failed to load chunks:", e)
if not CHUNKS:
    print("⚠️ No processed_chunks.json found; running without local context.")

# -----------------------------
# Date parsing (German)
# -----------------------------
DE_MONTHS = {
    'januar': 1, 'jan': 1,
    'februar': 2, 'feb': 2,
    'maerz': 3, 'märz': 3, 'mrz': 3, 'maer': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'mai': 5,
    'juni': 6, 'jun': 6,
    'juli': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'oktober': 10, 'okt': 10,
    'november': 11, 'nov': 11,
    'dezember': 12, 'dez': 12
}

def _normalize_dash(s: str) -> str:
    return s.replace("\u2013", "-").replace("\u2014", "-").replace("–","-").replace("—","-")

def parse_de_date_text(txt: str) -> Optional[datetime]:
    if not txt:
        return None
    s = _normalize_dash(txt.lower().strip())
    s = s.replace("uhr", "").replace(",", " ")
    m = re.search(r"(\d{1,2})\s*(?:\.\s*)?(jan|januar|feb|februar|maerz|märz|mrz|maer|mar|april|apr|mai|jun|juni|jul|juli|aug|august|sep|sept|september|okt|oktober|nov|november|dez|dezember)\s*(\d{4})", s)
    if m:
        day = int(m.group(1))
        mon = m.group(2)
        month = DE_MONTHS.get(mon, None)
        year = int(m.group(3))
    else:
        m2 = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
        if not m2:
            return None
        day, month, year = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
    tmatch = re.search(r"(\d{1,2}):(\d{2})", s)
    hh, mm = (0, 0)
    if tmatch:
        hh, mm = int(tmatch.group(1)), int(tmatch.group(2))
    try:
        return datetime(year, month, day, hh, mm, tzinfo=timezone.utc)
    except Exception:
        return None

# -----------------------------
# Live: Impuls-Workshops
# -----------------------------
def dedupe_items(items, key=lambda x: (x.get('title','').lower().strip(), x.get('when',''))):
    seen = set()
    out = []
    for it in items:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out

def get_upcoming_impuls_workshops_live(max_items: int = 10) -> List[Dict]:
    url = "https://dlh.zh.ch/home/impuls-workshops"
    print("LIVE FETCH: Fetching current Impuls-Workshops page")
    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": "DLH-Chatbot/1.0"})
        resp.raise_for_status()
    except Exception as e:
        print("LIVE FETCH ERROR (Impuls):", e)
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    candidates = soup.select("li, .event, .teaser, .item, article")
    events = []
    now = datetime.now(timezone.utc)
    for el in candidates:
        a = el.find("a")
        title = a.get_text(strip=True) if a else el.get_text(" ", strip=True)[:120]
        href = a.get("href") if a and a.has_attr("href") else url
        full_url = urllib.parse.urljoin(url, href)

        dt_el = None
        for css in ["time", ".date", ".datetime", ".termine", ".event-date"]:
            dt_el = el.select_one(css) if hasattr(el, "select_one") else None
            if dt_el:
                break
        when_text = (dt_el.get_text(" ", strip=True) if dt_el else el.get_text(" ", strip=True))
        when_text = _normalize_dash(when_text)
        dt = parse_de_date_text(when_text)
        if not dt or dt < now:
            continue

        desc_el = None
        for css in [".intro", ".desc", "p"]:
            desc_el = el.select_one(css) if hasattr(el, "select_one") else None
            if desc_el:
                break
        snippet = (desc_el.get_text(" ", strip=True) if desc_el else "")

        if re.search(r"impuls|workshop|reihe|mintwoch|one change", (title + " " + when_text).lower()):
            events.append({
                "title": title,
                "url": full_url,
                "when": dt.isoformat(),
                "when_text": when_text,
                "snippet": snippet
            })
    events = dedupe_items(events)
    events.sort(key=lambda e: e["when"])
    print(f"LIVE FETCH SUCCESS (Impuls): found {len(events)} future events")
    return events[:max_items]

def fetch_live_impuls_workshops() -> Optional[Dict]:
    events = get_upcoming_impuls_workshops_live(max_items=12)
    if not events:
        return None
    lines = ["<ul>"]
    for e in events:
        title_html = f'<a href="{e["url"]}" target="_blank" rel="noopener">{e["title"]}</a>'
        li = f'<li><strong>{e["when_text"]}</strong> – {title_html}</li>'
        lines.append(li)
    lines.append("</ul>")
    content = "\n".join(lines)
    return {
        "content": content,
        "metadata": {
            "source": "https://dlh.zh.ch/home/impuls-workshops",
            "title": "Impuls-Workshops - Digital Learning Hub Sek II (LIVE)",
            "is_event_page": True,
            "fetched_live": True
        }
    }

# -----------------------------
# Live: Innovationsfonds per subject
# -----------------------------
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

def fetch_live_innovationsfonds(subject: Optional[str] = None) -> Optional[Dict]:
    base = "https://dlh.zh.ch"
    if subject:
        key = subject.lower()
        slug = SUBJECT_SLUGS.get(key, key)
        url = f"{base}/home/innovationsfonds/projektvorstellungen/uebersicht/filterergebnisse-fuer-projekte/tags/{slug}"
    else:
        url = f"{base}/home/innovationsfonds/projektvorstellungen/uebersicht"
    print(f"LIVE FETCH: Innovationsfonds projects for subject='{subject or 'overview'}' from {url}")
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "DLH-Chatbot/1.0"})
        r.raise_for_status()
    except Exception as e:
        print("LIVE FETCH ERROR (Innovationsfonds):", e)
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    candidates = soup.select("article a, .card a, .teaser a, li a")
    projects = []
    for a in candidates:
        href = a.get("href")
        title = a.get_text(strip=True)
        if not href or not title:
            continue
        full = href if href.startswith("http") else base + href
        if "/innovationsfonds/" in full and "/tags/" not in full and "/uebersicht" not in full:
            projects.append({"title": title, "url": full})
        if len(projects) >= 12:
            break
    if not projects:
        print("LIVE FETCH: No project cards detected on the page")
        return None
    out = ["<ul>"]
    for p in projects:
        out.append(f'<li><a href="{p["url"]}" target="_blank" rel="noopener">{p["title"]}</a></li>')
    out.append("</ul>")
    content = "\n".join(out)
    print(f"LIVE FETCH SUCCESS (Innovationsfonds): Compiled {len(projects)} projects")
    return {
        "content": content,
        "metadata": {
            "source": url,
            "title": f"Innovationsfonds Projekte - {subject or 'Uebersicht'} (LIVE)",
            "fetched_live": True
        }
    }

# -----------------------------
# Intent / search
# -----------------------------
def normalize_query(q: str) -> Tuple[str, str]:
    ql = q.lower()
    q_norm = ql.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    return ql, q_norm

def extract_query_intent(query: str) -> Dict:
    query_lower, q_norm = normalize_query(query)

    subjects = {
        "chemie": ["chemie"],
        "physik": ["physik"],
        "biologie": ["biologie"],
        "mathematik": ["mathematik", "mathe"],
        "informatik": ["informatik", "cs"],
        "deutsch": ["deutsch"],
        "englisch": ["englisch", "english"],
        "franzoesisch": ["franzoesisch"],
        "italienisch": ["italienisch"],
        "spanisch": ["spanisch"],
        "geschichte": ["geschichte"],
        "geografie": ["geografie", "geographie"],
        "wirtschaft": ["wirtschaft", "w&r", "wr"],
        "recht": ["recht"],
        "philosophie": ["philosophie"],
    }

    subject_keywords = [key for key, kws in subjects.items() if any(kw in q_norm for kw in kws)]
    topic_keywords = [w for w in ["impuls", "workshop", "veranstaltung", "termine", "events", "innovationsfonds", "projekt", "projekte"] if w in query_lower]

    is_date_query = any(w in query_lower for w in ["wann", "nächsten", "kommenden", "termine", "wann sind", "welche workshops", "welches sind die nächsten"])
    is_innovationsfonds_query = ("innovationsfonds" in query_lower) or ("innovations" in query_lower) or ("projekt" in query_lower) or ("projekte" in query_lower)

    return {
        "query_lower": query_lower,
        "q_norm": q_norm,
        "subject_keywords": subject_keywords,
        "topic_keywords": topic_keywords,
        "is_date_query": is_date_query,
        "is_innovationsfonds_query": is_innovationsfonds_query,
    }

def advanced_search(query: str, max_items: int = 8) -> List[Tuple[int, Dict]]:
    intent = extract_query_intent(query)
    query_lower = intent["query_lower"]
    results: List[Tuple[int, Dict]] = []

    # Workshops live
    if intent["is_date_query"] or any(k in query_lower for k in ["impuls", "workshop", "termine", "veranstaltung", "events"]):
        live_chunk = fetch_live_impuls_workshops()
        if live_chunk:
            results.append((220, live_chunk))

    # Innovationsfonds live for any detected subject
    if intent["subject_keywords"]:
        for subj in intent["subject_keywords"]:
            live_proj = fetch_live_innovationsfonds(subject=subj)
            if live_proj:
                results.append((300, live_proj))

    # Fallback: cached chunks
    for ch in CHUNKS[:256]:
        results.append((120, ch))

    # sort & dedupe
    results.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    filtered: List[Tuple[int, Dict]] = []
    for score, ch in results:
        meta = ch.get("metadata", {})
        key = (meta.get("source", ""), meta.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        filtered.append((score, ch))
    return filtered[:max_items]

# -----------------------------
# Prompt building
# -----------------------------
def build_system_prompt() -> str:
    return (
        "Du bist der offizielle DLH Chatbot. Antworte auf Deutsch mit HTML-Formatierung. "
        "Wenn der Kontext mehrere Termin- oder Projektzeilen enthaelt, liste ALLE als eine HTML-Liste "
        "(<ul><li>…</li></ul>) auf. Nenne bei Terminen Datum und Zeit sowie einen verlinkten Titel. "
        "Bei Projekten im Innovationsfonds liste die Titel jeweils als klickbare Links."
    )

def build_user_prompt(question: str, ranked_chunks: List[Tuple[int, Dict]]) -> str:
    lines = [f"FRAGE: {question}", "", "KONTEXT:"]
    for score, ch in ranked_chunks:
        meta = ch.get("metadata", {})
        src = meta.get("source", "")
        title = meta.get("title", "")
        content = ch.get("content", "")
        if len(content) > 3500:
            content = content[:3500] + "…"
        lines.append(f"[{score}] {title} <{src}>")
        lines.append(content)
        lines.append("")
    lines.append("AUFGABE: Antworte präzise und knapp. Bei Termin- oder Projektlisten verwende HTML-Listen.")
    return "\n".join(lines)

# -----------------------------
# LLM call
# -----------------------------
def call_openai(system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str:
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=max_tokens,   # ✅ richtig
        # KEIN temperature / top_p / frequency_penalty ...
    )
    return resp.choices[0].message.content.strip()
# -----------------------------
# Sources builder
# -----------------------------
def build_sources(ranked: List[Tuple[int, Dict]], limit: int = 3) -> List[Dict]:
    out = []
    for score, ch in ranked:
        meta = ch.get("metadata", {})
        url = meta.get("source", "")
        title = meta.get("title", "") or "Quelle"
        if url:
            out.append({
                "url": url,
                "title": title,
                "snippet": ch.get("content", "")[:180]
            })
        if len(out) >= limit:
            break
    return out

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "healthy", "chunks_loaded": len(CHUNKS), "model": OPENAI_MODEL}

@app.get("/version")
def version():
    return {"version": "openai-backend", "model": OPENAI_MODEL}
    
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "DLH OpenAI API",
        "endpoints": ["/health", "/ask", "/version"]    
    }
@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    ranked = advanced_search(req.question, max_items=req.max_sources or 8)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(req.question, ranked)
    answer = call_openai(system_prompt, user_prompt, max_tokens=1200)
    sources = build_sources(ranked, limit=req.max_sources or 3)
    return AnswerResponse(answer=answer, sources=sources)

# -----------------------------
# Local dev
# -----------------------------


# === Functions carried over from old live_patch2_fix (needed) ===

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

def load_and_preprocess_data():
    """Lade und bereite Daten mit verbesserter Struktur vor"""
    try:
        file_path = 'processed/processed_chunks.json'
        
        print(f"Y Attempting to load data from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f" Successfully loaded {len(chunks)} chunks from {file_path}")
        
        # Erstelle Index fuer schnellere Suche
        keyword_index = {}
        url_index = {}
        subject_index = {}
        
        for i, chunk in enumerate(chunks):
            # URL-basierter Index
            url = chunk['metadata'].get('source', '').lower()
            if url not in url_index:
                url_index[url] = []
            url_index[url].append(i)
            
            # FAcher-Index aus Metadaten
            faecher = chunk['metadata'].get('faecher', [])
            if faecher:
                for fach in faecher:
                    fach_lower = fach.lower()
                    if fach_lower not in subject_index:
                        subject_index[fach_lower] = []
                    subject_index[fach_lower].append(i)
            
            # Keyword-Index
            content = chunk['content'].lower()
            important_terms = [
                'fobizz', 'genki', 'innovationsfonds', 'cop', 'cops',
                'vernetzung', 'workshop', 'weiterbildung', 'kuratiert',
                'impuls', 'termin', 'anmeldung', 'lunch', 'learn',
                'impuls-workshop', 'impulsworkshop', 'veranstaltung', 'event',
                'chemie', 'physik', 'biologie', 'mathematik', 'informatik',
                'deutsch', 'englisch', 'franzAsisch', 'italienisch', 'spanisch',
                'geschichte', 'geografie', 'wirtschaft', 'recht', 'philosophie'
            ]
            
            for term in important_terms:
                if term in content:
                    if term not in keyword_index:
                        keyword_index[term] = []
                    keyword_index[term].append(i)
        
        print(f"Y Indexed {len(keyword_index)} keywords")
        print(f"Ys Indexed {len(subject_index)} subjects in metadata")
        
        return chunks, keyword_index, url_index, subject_index
    except Exception as e:
        print(f"a Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return [], {}, {}, {}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ultimate_api:app", host="0.0.0.0", port=8000)
