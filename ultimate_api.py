import os
import json
import re
import urllib.parse
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from traceback import format_exc

from pydantic_settings import BaseSettings
from pydantic import BaseModel, ValidationError

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
# Load knowledge base chunks, support .json and .jsonl formats
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "processed/processed_chunks.json")

def call_openai(system_prompt, user_prompt, max_tokens=1200):
    """
    Calls the OpenAI API with the specified system and user prompts.
    Returns the response text.
    """
    response = openai_client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=max_tokens,
        temperature=0.3,
        stream=False,
    )
    return response.choices[0].message.content.strip()


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
PROMPT_CHARS_BUDGET = int(os.getenv("PROMPT_CHARS_BUDGET", "24000"))

# optionale Feintuning-Parameter
MAX_HITS_IN_PROMPT   = 12   # höchstens so viele Treffer einbetten
MAX_SNIPPET_CHARS    = 800  # pro Treffer; wird vor dem Einfügen gekürzt
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "processed/processed_chunks.json")

# Datei laden
CHUNKS: list[dict] = load_chunks(CHUNKS_PATH)
CHUNKS_COUNT = len(CHUNKS)
logger.info(f"✅ Loaded {CHUNKS_COUNT} chunks from {CHUNKS_PATH}")

# -----------------------------------------------------------------
# Sitemap loader + simple section index
# -----------------------------------------------------------------
SITEMAP_URLS: list[str] = []
SITEMAP_SECTIONS: dict[str, list[str]] = {}
SITEMAP_LOADED = False

def load_sitemap(local_path: str = "processed/dlh_sitemap.xml") -> dict[str, int]:
    """
    Lädt eine Standard-XML-Sitemap, indexiert URLs und einfache Sektions-Buckets.
    """
    global SITEMAP_URLS, SITEMAP_SECTIONS, SITEMAP_LOADED
    stats = {"urls": 0, "sections": 0, "ok": 0}
    try:
        p = Path(local_path)
        if not p.exists():
            print("Sitemap not found at", local_path)
            return stats
        tree = ET.parse(str(p))
        root = tree.getroot()
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls: list[str] = []
        for u in root.findall("sm:url", ns):
            loc = u.findtext("sm:loc", default="", namespaces=ns).strip()
            if loc:
                urls.append(loc)

        buckets: dict[str, list[str]] = {}
        KEYS = [
            "impuls-workshops", "innovationsfonds", "genki", "vernetzung",
            "weiterbildung", "kuratiertes", "cops", "wb-kompass", "fobizz", "schulalltag"
        ]
        for u in urls:
            path = urllib.parse.urlparse(u).path.lower()
            for k in KEYS:
                if f"/{k}" in path:
                    buckets.setdefault(k, []).append(u)

        SITEMAP_URLS = urls
        SITEMAP_SECTIONS = buckets
        SITEMAP_LOADED = True
        stats.update({"urls": len(urls), "sections": len(buckets), "ok": 1})
        return stats
    except Exception as e:
        print("WARN: sitemap load failed:", repr(e))
        return stats

def sitemap_candidates_for_query(q: str, limit: int = 6) -> list[dict]:
    """
    Liefert priorisierte Kandidaten-URLs aus der Sitemap passend zur Query.
    Formatiert als 'fake hits' wie aus dem Index (title/url/snippet/metadata.source).
    """
    if not SITEMAP_LOADED or not q:
        return []
    ql = q.lower()
    hits: list[str] = []

    if any(k in ql for k in ["impuls", "workshop"]):
        hits += SITEMAP_SECTIONS.get("impuls-workshops", [])
    if "innovationsfonds" in ql or "innovations" in ql:
        hits += SITEMAP_SECTIONS.get("innovationsfonds", [])
    if "genki" in ql:
        hits += SITEMAP_SECTIONS.get("genki", [])
    if "cops" in ql or "community" in ql:
        hits += SITEMAP_SECTIONS.get("cops", [])
    if "weiterbildung" in ql:
        hits += SITEMAP_SECTIONS.get("weiterbildung", [])
    if "kuratiert" in ql or "kuratiertes" in ql:
        hits += SITEMAP_SECTIONS.get("kuratiertes", [])

    seen = set()
    out: list[dict] = []
    for u in hits:
        if u in seen:
            continue
        seen.add(u)
        title_guess = urllib.parse.urlparse(u).path.rsplit("/", 1)[-1].replace("-", " ").strip().title() or "DLH Seite"
        out.append({
            "title": title_guess,
            "url": u,
            "snippet": "",
            "metadata": {"source": u}
        })
        if len(out) >= limit:
            break
    return out

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
    allow_origins=["https://perino.info"],  # Adjust to your needs
    allow_credentials=False,  # False if allow_origins=["*"]
    allow_methods=["GET", "POST", "OPTIONS",],
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

# Root endpoint for basic check
@app.get("/")
async def root():
    return {"message": "DLH Chatbot API is up"}

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}
    
@app.get("/debug/impuls")
def debug_impuls():
    ev = fetch_live_impuls_workshops()
    return {"count": len(ev), "sample": ev[:3]}
    
def render_workshops_html(items: List[Dict]) -> str:
    """Erzeugt eine kompakte HTML-Timeline mit klickbaren Titeln."""
    # Weiterleitung auf die neue Timeline-Version
    return render_workshops_timeline_html(items)


def normalize_subject_to_slug(text: str) -> Optional[str]:
    """
    Verwendet die globale SUBJECT_SLUGS-Tabelle, um ein Fach
    aus der Benutzerfrage dem passenden Tag-Slug zuzuordnen.
    """
    if not text:
        return None
    t = text.lower()
    for key, slug in SUBJECT_SLUGS.items():
        if key in t:
            return slug
    return None
# Main ask endpoint (example impl)
@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    try:
        ranked = get_ranked_with_sitemap(req.question, max_items=req.max_sources or 12)
        q_low = req.question.lower() if req.question else ""
        
        # Direct early handling for Impuls-Workshop questions
        if any(k in q_low for k in ["impuls", "workshop", "workshops"]):
            events = fetchliveimpulsworkshops()
            html = render_workshopstimeline_html(events, title="Kommende Impuls-Workshops")
            sources = [SourceItem(title="Impuls-Workshop-Übersicht", url="https://dlh.zh.ch/home/impuls-workshops")]
            return AnswerResponse(answer=html, sources=sources)
        
        # General LLM workflow
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(req.question, ranked)
        answer_html = ensure_clickable_links(call_openai(system_prompt, user_prompt, max_tokens=1200))
        sources = build_sources(ranked, limit=req.max_sources or 4)
        return AnswerResponse(answer=answer_html, sources=sources)
    except Exception as e:
        logger.error("ERROR /ask", exc_info=e)
        msg = "Entschuldigung, es gab einen technischen Fehler. Bitte versuchen Sie es später erneut."
        return AnswerResponse(answer=msg, sources=[])


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
def sitemap_find_innovations_tag(tag_slug: str) -> Optional[str]:
    """
    Liefert die URL der Innovationsfonds-Tag-Seite aus der Sitemap,
    z. B. tag_slug='chemie' → .../innovationsfonds/.../tags/chemie
    """
    if not SITEMAP_LOADED or not tag_slug:
        return None
    for u in SITEMAP_URLS:
        p = urllib.parse.urlparse(u).path.lower()
        if "/innovationsfonds" in p and "/tags/" in p and p.endswith(f"/{tag_slug}"):
            return u
    return None


    # 1) Aus Sitemap
    if SITEMAP_LOADED:
        for u in SITEMAP_URLS:
            p = urllib.parse.urlparse(u).path.lower()
            if "/innovationsfonds" in p and "/tags/" in p and p.endswith(f"/{tag_slug}"):
                return u

    # 2) Fallback (bekannte DLH-Struktur)
    fallback = f"https://dlh.zh.ch/home/innovationsfonds/projektvorstellungen/uebersicht/filterergebnisse-fuer-projekte/tags/{tag_slug}"
    return fallback

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
def ensure_clickable_links(html_text: str) -> str:
    """
    Wandelt nackte URLs im Text in anklickbare Links um:
    https://beispiel -> <a href='...' target='_blank'>...</a>
    """
    url_re = re.compile(r'(https?://[^\s<>"\)]+)')
    def repl(m):
        u = m.group(1)
        return f"<a href='{html.escape(u)}' target='_blank'>{html.escape(u)}</a>"
    return url_re.sub(repl, html_text)

import inspect

REQUIRED_SYS_HINTS = [
    "valide", "HTML", "Quellen", "<a href", "Liste", "Timeline"  # locker gehalten
]

def _sample_hits():
    # Minimale, realistische Testdaten für den Prompt
    return [
        {
            "title": "Impulsworkshops – Übersicht",
            "url": "https://dlh.zh.ch/home/impuls-workshops",
            "snippet": "Aktuelle und kommende Impuls-Workshops mit Datum und Anmeldung."
        },
        {
            "title": "Innovationsfonds – Chemie",
            "url": "https://dlh.zh.ch/home/innovationsfonds/chemie",
            "snippet": "Sechs Projekte im Fach Chemie mit Kurzbeschreibung."
        }
    ]

def validate_prompts() -> dict:
    """Prüft System-/User-Prompt, Link-Formatter und call_openai-Signatur."""
    results = {"ok": True, "checks": []}

    # 1) build_system_prompt vorhanden & enthält die erwarteten Stichworte
    try:
        sp = build_system_prompt()
        ok = isinstance(sp, str) and len(sp) > 20 and all(h.lower() in sp.lower() for h in REQUIRED_SYS_HINTS)
        results["checks"].append({"name": "build_system_prompt", "ok": ok, "len": len(sp)})
        if not ok:
            results["ok"] = False
    except Exception as e:
        results["checks"].append({"name": "build_system_prompt", "ok": False, "error": repr(e)})
        results["ok"] = False

    # 2) build_user_prompt erzeugt sinnvollen Text mit Frage+Treffern
    try:
        up = build_user_prompt("Welche Impuls-Workshops stehen als Nächstes an?", _sample_hits())
        ok = isinstance(up, str) and "Benutzerfrage" in up and "Relevante Auszüge" in up and "http" in up
        results["checks"].append({"name": "build_user_prompt", "ok": ok, "len": len(up)})
        if not ok:
            results["ok"] = False
    except Exception as e:
        results["checks"].append({"name": "build_user_prompt", "ok": False, "error": repr(e)})
        results["ok"] = False

    # 3) ensure_clickable_links macht aus URL einen <a>-Link
    try:
        test_html = "Siehe https://dlh.zh.ch/home/impuls-workshops für Details."
        html_out = ensure_clickable_links(test_html)
        ok = "<a href=" in html_out and "target='_blank'" in html_out
        results["checks"].append({"name": "ensure_clickable_links", "ok": ok})
        if not ok:
            results["ok"] = False
    except Exception as e:
        results["checks"].append({"name": "ensure_clickable_links", "ok": False, "error": repr(e)})
        results["ok"] = False

    # 4) call_openai Signatur: (system_prompt, user_prompt, max_tokens=...)
    try:
        sig = inspect.signature(call_openai)
        params = list(sig.parameters.keys())
        ok = params[:2] == ["system_prompt", "user_prompt"] and "max_tokens" in params
        results["checks"].append({"name": "call_openai_signature", "ok": ok, "params": params})
        if not ok:
            results["ok"] = False
    except Exception as e:
        results["checks"].append({"name": "call_openai_signature", "ok": False, "error": repr(e)})
        results["ok"] = False

    return results
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

def render_workshops_timeline_html(events: List[Dict], title: str = "Kommende Impuls-Workshops") -> str:
    if not events:
        return (
            "<section class='dlh-answer'>"
            "<p>Keine Workshops gefunden.</p>"
            "<h3>Quellen</h3>"
            "<ul class='sources'>"
            "<li><a href='https://dlh.zh.ch/home/impuls-workshops' target='_blank'>Impuls-Workshop-Übersicht</a></li>"
            "</ul></section>"
        )

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

    return (
        "<section class='dlh-answer'>"
        f"<p>{title}:</p>"
        "<ol class='timeline'>"
        + "".join(lis) +
        "</ol>"
        "<h3>Quellen</h3>"
        "<ul class='sources'>"
        "<li><a href='https://dlh.zh.ch/home/impuls-workshops' target='_blank'>Impuls-Workshop-Übersicht</a></li>"
        "</ul></section>"
    )
    cards = []
    for it in items:
        title = it.get("title", "(ohne Titel)")
        url = it.get("url", "#")
        snip = (it.get("snippet") or "").strip()
        cards.append(
            "<article class='card'>"
            f"  <h4><a href='{url}' target='_blank'>{title}</a></h4>"
            f"  <p>{snip}</p>"
            "</article>"
        )

    html = (
        "<section class='dlh-answer'>"
        f"  <p>Innovationsfonds-Projekte im Fach <strong>{subject_title}</strong>:</p>"
        f"  <div class='cards'>{''.join(cards)}</div>"
        "  <h3>Quellen</h3>"
        "  <ul class='sources'>"
        f"    <li><a href='{tag_url}' target='_blank'>Tag-Seite: {subject_title}</a></li>"
        "  </ul>"
        "</section>"
    )
    return html
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
