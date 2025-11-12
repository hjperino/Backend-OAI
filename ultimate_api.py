import os
import json
import re
import urllib.parse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    settings = Settings()
    logger.info("...")
except ValidationError:
    logger.critical("...")

from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from traceback import format_exc

from pydantic_settings import BaseSettings
from pydantic import ValidationError
from pydantic import BaseModel

from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic environment config
class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-5"
    chunks_path: str
    # Optionally add openai_org_id if needed:
    # openai_org_id: Optional[str] = None

try:
    settings = Settings()
    logger.info(f"OpenAI model: {settings.openai_model}, chunks path: {settings.chunks_path}")
except ValidationError as e:
    logger.critical(f"Environment settings validation failed: {e.json()}")
    raise

# Initialize OpenAI client
openai_client = OpenAI(
    api_key=settings.openai_api_key,
    # organization=settings.openai_org_id,  # Uncomment and add field if needed
)
PROMPT_CHARS_BUDGET = int(os.getenv("PROMPT_CHARS_BUDGET", "24000"))

# optionale Feintuning-Parameter
MAX_HITS_IN_PROMPT   = 12   # höchstens so viele Treffer einbetten
MAX_SNIPPET_CHARS    = 800  # pro Treffer; wird vor dem Einfügen gekürzt
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "processed/processed_chunks.json")

# Datei laden
CHUNKS: list[dict] = load_chunks(CHUNKS_PATH)
CHUNKS_COUNT = len(CHUNKS)
logger.info(f"✅ Loaded {CHUNKS_COUNT} chunks from {CHUNKS_PATH}")


def get_ranked_with_sitemap(query: str, max_items: int = 12) -> list[dict]:
    """
    Kombiniert Sitemap-Kandidaten (Boost) mit der bestehenden advanced_search.
    Verändert advanced_search nicht; fügt nur eine Boost-Schicht davor.
    """
    try:
        boosted = sitemap_candidates_for_query(query, limit=6)
    except Exception:
        boosted = []
    try:
        core = advanced_search(query, max_items=max_items)  # nutzt deine bestehende Funktion
    except Exception:
        core = []
    seen = set()
    merged: list[dict] = []
    def key(h):
        if isinstance(h, dict):
            return h.get("url") or h.get("metadata", {}).get("source")
        if isinstance(h, tuple) and len(h) >= 2 and isinstance(h[1], dict):
            hh = h[1]
            return hh.get("url") or hh.get("metadata", {}).get("source")
        return None
    for h in boosted + core:
        u = key(h)
        if not u or u in seen:
            continue
        seen.add(u)
        merged.append(h if isinstance(h, dict) else h[1])
        if len(merged) >= max_items:
            break
    return merged
    
# FastAPI application setup
app = FastAPI(title="DLH OpenAI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your needs
    allow_credentials=False,  # False if allow_origins=["*"]
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models for API
class SourceItem(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []

class QuestionRequest(BaseModel):
    question: str
    language: Optional[str] = "de"
    max_sources: Optional[int] = 3

# Load knowledge base chunks, support .json and .jsonl formats
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "processed/processed_chunks.json")

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
            return []

# Datei laden
CHUNKS: list[dict] = load_chunks(CHUNKS_PATH)
CHUNKS_COUNT = len(CHUNKS)
logger.info(f"✅ Loaded {CHUNKS_COUNT} chunks from {CHUNKS_PATH}")

CHUNKS: List[Dict] = load_chunks(settings.chunks_path)
logger.info(f"✅ Loaded {len(CHUNKS)} chunks from {settings.chunks_path}")

# Part 2 of ultimate_api-Kopie.py

MONTHS_DE = {
    "januar": 1, "jan": 1,
    "februar": 2, "feb": 2,
    "märz": 3, "maerz": 3, "mrz": 3,
    "april": 4, "apr": 4,
    "mai": 5,
    "juni": 6, "jun": 6,
    "juli": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "oktober": 10, "okt": 10,
    "november": 11, "nov": 11,
    "dezember": 12, "dez": 12,
}

TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:\.]([0-5]\d)\b")
DMY_DOTTED_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b")
DMY_TEXT_RE = re.compile(
    r"\b(\d{1,2})\.\s*([A-Za-zÄÖÜäöüß]+)\s*(20\d{2}|\d{2})?\b",
    re.IGNORECASE,
)

def _coerce_year(y: Optional[str], ref_year: int) -> int:
    if not y:
        return ref_year
    y = y.strip()
    if len(y) == 2:
        y = "20" + y
    try:
        yi = int(y)
        return yi
    except Exception:
        return ref_year

def parse_de_date(text: str, ref_date: Optional[datetime] = None) -> Optional[datetime]:
    """
    Extracts a date (with optional time) from German text.
    Returns UTC-based datetime or None.
    Supports:
    - 11.11.2025 or 11.11.25
    - 25. Nov 2025 or 25. November
    - optional time like 16:30 or 16.30
    """
    if not text:
        return None
    t = text.strip()
    rd = ref_date or datetime.now(timezone.utc)
    ref_year = rd.year

    # Time extraction
    hh, mm = None, None
    tm = TIME_RE.search(t)
    if tm:
        hh, mm = int(tm.group(1)), int(tm.group(2))

    # Pattern 1: dotted date 11.11.2025
    m = DMY_DOTTED_RE.search(t)
    if m:
        try:
            d, mth, y = int(m.group(1)), int(m.group(2)), _coerce_year(m.group(3), ref_year)
            base = datetime(y, mth, d, tzinfo=timezone.utc)
            if hh is not None:
                base = base.replace(hour=hh, minute=mm or 0)
            return base
        except Exception:
            pass

    # Pattern 2: textual month date (e.g. 25. Nov 2025)
    m = DMY_TEXT_RE.search(t)
    if m:
        try:
            d = int(m.group(1))
            month_word = (m.group(2) or "").strip().lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
            mth = MONTHS_DE.get(month_word)
            y = _coerce_year(m.group(3), ref_year)
            if mth:
                base = datetime(y, mth, d, tzinfo=timezone.utc)
                if hh is not None:
                    base = base.replace(hour=hh, minute=mm or 0)
                return base
        except Exception:
            pass

    return None

from datetime import date as _date

def parse_de_date_to_date(text: str) -> Optional[_date]:
    """
    Helper to parse German date string and return datetime.date.
    """
    dt = parse_de_date(text)
    return dt.date() if dt else None
    
# Part 3 of ultimate_api-Kopie.py

from typing import Set

def dedupe_items(
    items: List[Dict],
    key=lambda x: (x.get("title", "").lower().strip(), x.get("when", ""))
) -> List[Dict]:
    seen: Set = set()
    out: List[Dict] = []
    for it in items:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out

def fetch_live_impuls_workshops() -> List[Dict]:
    """
    Fetches normalized live Impuls-Workshop events.
    Returns list of dicts with 'date', 'title', 'url', 'when', '_d' (date only) keys.
    """
    IMPULS_URL = "https://dlh.zh.ch/home/impuls-workshops"
    UA = {"User-Agent": "DLH-Chatbot/1.0 (+https://dlh.zh.ch)"}

    try:
        r = requests.get(IMPULS_URL, timeout=20, headers=UA, allow_redirects=True)
        r.raise_for_status()
        html = r.text
    except requests.RequestException as ex:
        logger.error("LIVE FETCH ERROR (Impuls, GET): %s", repr(ex))
        return []

    try:
        # Try faster parser first
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        # Fallback parser if lxml not available
        soup = BeautifulSoup(html, "html.parser")

    # Remove boilerplate elements
    for selector in [
        "script", "style", "noscript", ".cookie", ".consent", ".banner",
        "header", "footer", "nav", "aside"
    ]:
        for el in soup.select(selector):
            el.decompose()

    root = soup.select_one("main") or soup
    events_raw: List[Dict] = []

    # Parse lists: ol/ul > li, preferably with <a>, else plain text
    for li in root.select("ol li, ul li"):
        txt = li.get_text(" ", strip=True)
        if not txt:
            continue
        a = li.find("a")
        title = a.get_text(" ", strip=True) if a else txt
        href = a.get("href") if (a and a.has_attr("href")) else ""
        if href and href.startswith("/"):
            href = urllib.parse.urljoin(IMPULS_URL, href)
        time_el = li.find("time")
        when = time_el.get_text(" ", strip=True) if time_el else txt
        events_raw.append({"title": title, "url": href or IMPULS_URL, "when": when})

    # Fallback: look under headings with "Impuls"
    if not events_raw:
        heads = [h for h in root.select("h2, h3") if "impuls" in h.get_text(" ", strip=True).lower()]
        for h in heads:
            sec = h.find_next(["ol", "ul", "section", "div"]) or root
            for li in sec.select("li"):
                txt = li.get_text(" ", strip=True)
                if not txt:
                    continue
                a = li.find("a")
                title = a.get_text(" ", strip=True) if a else txt
                href = a.get("href") if (a and a.has_attr("href")) else ""
                if href and href.startswith("/"):
                    href = urllib.parse.urljoin(IMPULS_URL, href)
                time_el = li.find("time")
                when = time_el.get_text(" ", strip=True) if time_el else txt
                events_raw.append({"title": title, "url": href or IMPULS_URL, "when": when})

    # Normalize and parse dates
    norm: List[Dict] = []
    for e in events_raw:
        dt = parse_de_date_to_date(e.get("when") or "") or parse_de_date_to_date(e.get("title") or "")
        if not dt:
            # fallback: try combined text
            all_txt = f"{e.get('title','')} {e.get('when','')}"
            dt = parse_de_date_to_date(all_txt)
        if not dt:
            continue  # skip events without date

        item = dict(e)
        item["date"] = dt  # datetime.date
        item["_d"] = dt
        norm.append(item)

    # Deduplicate by date + title
    seen = set()
    cleaned: List[Dict] = []
    for e in norm:
        key = (e["_d"].isoformat(), e["title"])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(e)

    cleaned.sort(key=lambda x: x["_d"])
    logger.info(f"LIVE FETCH SUCCESS (Impuls): parsed {len(cleaned)} events (raw {len(events_raw)})")
    return cleaned
    
# Part 4 of ultimate_api-Kopie.py

from typing import Union

def fetch_live_innovationsfonds(subject: Optional[str] = None) -> Optional[Dict]:
    base = "https://dlh.zh.ch"
    if subject:
        key = subject.lower()
        slug = SUBJECT_SLUGS.get(key, key)
        url = f"{base}/home/innovationsfonds/projektvorstellungen/uebersicht/filterergebnisse-fuer-projekte/tags/{slug}"
    else:
        url = f"{base}/home/innovationsfonds/projektvorstellungen/uebersicht"

    logger.info(f"LIVE FETCH Innovationsfonds projects for subject='{subject or 'overview'}' from {url}")
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "DLH-Chatbot/1.0"})
        r.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"LIVE FETCH ERROR Innovationsfonds: {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    candidates = soup.select("article a, .card a, .teaser a, li a")
    projects: List[Dict] = []

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
        logger.info("LIVE FETCH: No project cards detected on the page")
        return None

    content_lines = [""]
    for p in projects:
        content_lines.append(f'{p["title"]}')
        content_lines.append("")
    content = "\n".join(content_lines)

    logger.info(f"LIVE FETCH SUCCESS Innovationsfonds: Compiled {len(projects)} projects")
    return {
        "content": content,
        "metadata": {
            "source": url,
            "title": f"Innovationsfonds Projekte - {subject or 'Uebersicht'} (LIVE)",
            "fetched_live": True
        },
    }

# Utility to safely extract dict from either dict or (score, dict) tuple
def _as_dict(hit: Union[Dict, Tuple[int, Dict]]) -> Dict:
    if isinstance(hit, dict):
        return hit
    if isinstance(hit, tuple) and len(hit) > 1 and isinstance(hit[1], dict):
        return hit[1]
    return {}

def build_system_prompt() -> str:
    return (
        "Du bist der offizielle DLH Chatbot. Antworte auf Deutsch mit HTML-Formatierung. "
        "Wenn der Kontext mehrere Termin- oder Projektzeilen enthält, liste ALLE als eine HTML-Liste "
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

def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else (s[: max(0, n - 1)] + "…")

def build_user_prompt(question: str, hits: List[Dict]) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = [
        f"Heutiges Datum: {today}",
        f"Benutzerfrage: {question}",
        "",
        "Relevante Auszüge:",
    ]

    used = sum(len(p) for p in parts)  # current usage

    for h in hits[:MAX_HITS_IN_PROMPT]:
        title = h.get("title") or h.get("metadata", {}).get("title") or "Ohne Titel"
        url = h.get("url") or h.get("metadata", {}).get("source") or ""
        snippet = (
            h.get("snippet")
            or h.get("content")
            or h.get("metadata", {}).get("description")
            or ""
        )
        block = f"- {title} — {url}\n {_truncate(snippet, MAX_SNIPPET_CHARS)}\n"
        if used + len(block) > PROMPT_CHARS_BUDGET:
            break
        parts.append(block)
        used += len(block)

    parts.append(
        "Aufgabe: Antworte in sauberem HTML, wie im System-Prompt beschrieben. "
        "Verwende Listen (…) oder Karten, wenn sinnvoll. "
        "Verlinke Quellen mit Titel."
    )

    return "\n".join(parts)

# Part 5 of ultimate_api-Kopie.py

import html
import inspect

REQUIRED_SYS_HINTS = ["valide", "HTML", "Quellen"]

def ensure_clickable_links(html_text: str) -> str:
    """
    Wandelt nackte URLs im Text in anklickbare Links um.
    """
    url_re = re.compile(r'(https?://[^\s<>"\)]+)')
    def repl(m):
        u = m.group(1)
        return f'<a href="{html.escape(u)}" target="_blank">{html.escape(u)}</a>'
    return url_re.sub(repl, html_text)

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    try:
        ranked = get_ranked_with_sitemap(req.question, max_items=req.max_sources or 12)
        logger.debug(f"Y ranked types: {[type(x).__name__ for x in ranked[:5]]}")

        q_low = (req.question or "").lower()

        if any(k in q_low for k in ["impuls", "workshop", "workshops"]):
            # Handle workshop early exit with live scraping and direct rendering
            def _norm(s: str) -> str:
                s = (s or "").lower()
                return s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
            qn = _norm(req.question)
            live = fetch_live_impuls_workshops()
            if not live:
                live = fallback_events_from_chunks()

            want_past = any(k in qn for k in [
                "gab es", "waren", "vergangenen", "bisherigen", "im jahr", "letzten", "fruehere", "frühere", "bisher"
            ])
            want_next = any(k in qn for k in [
                "naechste", "nächste", "der naechste", "der nächste", "als naechstes", "als nächstes",
                "nur der naechste", "nur der nächste", "naechstes", "nächstes"
            ])
            yr = None
            m = re.search(r"(?:jahr|jahrgang|seit)\s*(20\d{2})", qn)
            if m:
                try:
                    yr = int(m.group(1))
                except Exception:
                    yr = None

            norm = []
            for e in live or []:
                d = _event_to_date(e)
                if d:
                    ee = dict(e)
                    ee["_d"] = d
                    norm.append(ee)

            today = datetime.now(timezone.utc).date()
            future = [e for e in norm if e.get("_d") and e["_d"] >= today]
            past = [e for e in norm if e.get("_d") and e["_d"] < today]

            if want_next:
                events_to_show = sorted(future, key=lambda x: x["_d"])[:1]
                html_str = render_workshops_timeline_html(events_to_show, title="Nächster Impuls-Workshop")
                return AnswerResponse(
                    answer=html_str,
                    sources=[SourceItem(title="Impuls-Workshop-Übersicht", url="https://dlh.zh.ch/home/impuls-workshops")]
                )
            if want_past:
                if yr:
                    past = [e for e in past if e["_d"].year == yr]
                events_to_show = sorted(past, key=lambda x: x["_d"], reverse=True)
                html_str = render_workshops_timeline_html(events_to_show, title=f"Vergangene Impuls-Workshops{(' ' + str(yr)) if yr else ''}")
                return AnswerResponse(
                    answer=html_str,
                    sources=[SourceItem(title="Impuls-Workshop-Übersicht", url="https://dlh.zh.ch/home/impuls-workshops")]
                )
            # Default: all future workshops
            events_to_show = sorted(future, key=lambda x: x["_d"])
            html_str = render_workshops_timeline_html(events_to_show, title="Kommende Impuls-Workshops")
            return AnswerResponse(
                answer=html_str,
                sources=[SourceItem(title="Impuls-Workshop-Übersicht", url="https://dlh.zh.ch/home/impuls-workshops")]
            )

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
                        html_str = render_innovationsfonds_cards_html(cards, subject_title=tag_slug.capitalize(), tag_url=tag_url)
                        srcs = [SourceItem(title=f"Innovationsfonds – {tag_slug}", url=tag_url, snippet=f"Projekte mit Tag {tag_slug}")]
                        if cards and cards[0].get("url"):
                            srcs.append(SourceItem(title=cards[0]["title"], url=cards[0]["url"], snippet=cards[0].get("snippet", "")))
                        return AnswerResponse(answer=html_str, sources=srcs)
        # Default LLM path
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(req.question, ranked)
        logger.debug(f"Y LLM call → {settings.openai_model} | prompt_len: {len(user_prompt)}")
        answer_html = call_openai(system_prompt, user_prompt, max_tokens=1200)
        answer_html = ensure_clickable_links(answer_html)
        sources = build_sources(ranked, limit=req.max_sources or 4)
        return AnswerResponse(answer=answer_html, sources=sources)
    except Exception as e:
        logger.error("ERROR /ask: %s\n%s", repr(e), format_exc())
        msg = "Entschuldigung, es gab einen technischen Fehler. Bitte versuchen Sie es später erneut."
        return AnswerResponse(answer=msg, sources=[])

@app.get("/debug/validate")
def debug_validate():
    """Runs prompt validation without calling OpenAI."""
    res = validate_prompts()
    return res

@app.get("/debug/impuls")
def debug_impuls():
    ev = fetch_live_impuls_workshops()
    return {"count": len(ev), "sample": ev[:3]}

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

# Add any remaining utility and validation functions here, using logging as appropriate.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ultimate_api:app", host="0.0.0.0", port=8000)
