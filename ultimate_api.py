# DLH Chatbot API (OpenAI) — full file
import os
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# --- Framework-Basis: FastAPI, Pydantic, CORS, Logging ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from traceback import format_exc

# --- OpenAI Client ---
# --- OpenAI Client ---
from openai import OpenAI
import openai as _openai_pkg
import bs4 as _bs4_pkg
import lxml as _lxml_pkg

# --- Dependency Versions (Debug-Ausgabe beim Start) ---
print(
    f"[DEPS] openai={getattr(_openai_pkg, '__version__', '?')}, "
    f"bs4={getattr(_bs4_pkg, '__version__', '?')}, "
    f"lxml={getattr(_lxml_pkg, '__version__', '?')}"
)
# --- OpenAI Konfiguration (vor späterer /ask-Route) ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID") or None,
)
app = FastAPI(title="DLH OpenAI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # oder: ["https://perino.info","https://www.perino.info"]
    allow_credentials=False,        # bei "*" muss das False sein
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Pydantic-Modelle ---
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

# --- OpenAI Konfiguration (vor späterer /ask-Route) ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID") or None,
)
# --- Prompt-Budget / Trimmen ---
# Gesamt-Budget in Zeichen für den User-Prompt (inkl. Snippets).
# Per ENV übersteuerbar: PROMPT_CHARS_BUDGET=24000
PROMPT_CHARS_BUDGET = int(os.getenv("PROMPT_CHARS_BUDGET", "24000"))

# optionale Feintuning-Parameter
MAX_HITS_IN_PROMPT   = 12   # höchstens so viele Treffer einbetten
MAX_SNIPPET_CHARS    = 800  # pro Treffer; wird vor dem Einfügen gekürzt
# -----------------------------
# Load processed chunks
# -----------------------------
# --- KB loading (supports .json and .jsonl) ---
# --- Knowledge Base loading (supports .json and .jsonl) ---
from pathlib import Path

CHUNKS_PATH = os.getenv("CHUNKS_PATH", "processed/processed_chunks.json")

def load_chunks(path: str):
    """Lädt die Wissensbasis (.json oder .jsonl) sicher und robust."""
    p = Path(path)
    if not p.exists():
        print(f"⚠️ KB not found at {p.resolve()}")
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
                    continue
        return out
    else:
        try:
            return json.load(p.open("r", encoding="utf-8"))
        except Exception as e:
            print("Failed to load chunks:", e)
            return []

# Datei laden
CHUNKS: list[dict] = load_chunks(CHUNKS_PATH)
CHUNKS_COUNT = len(CHUNKS)
print(f"✅ Loaded {CHUNKS_COUNT} chunks from {CHUNKS_PATH}")

# --- Komfort-Endpoints (optional, aber hilfreich) ---
@app.get("/health")
def health():
    # falls du CHUNKS o.ä. nutzt, hier die realen Zahlen einsetzen
    chunks_loaded = globals().get("CHUNKS_COUNT", 0)
    return {"status": "healthy", "chunks_loaded": chunks_loaded, "model": OPENAI_MODEL}

@app.get("/version")
def version():
    return {"version": "openai-backend", "model": OPENAI_MODEL}

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"ok": True, "service": "DLH OpenAI API", "endpoints": ["/health", "/ask", "/version"]}

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
def fetch_live_innovationsfonds_cards(tag_url: str, max_items: int = 12) -> List[Dict]:
    """
    Extrahiert Projektkarten (title, url, snippet) von der Tag-Seite.
    Nur echte Detailseiten; Menü/Tags/Übersichten werden ausgeschlossen.
    """
    try:
        r = requests.get(tag_url, timeout=12)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    main = soup.select_one("main") or soup

    # Kandidaten-Links
    links = main.find_all("a", href=True)
    items: List[Dict] = []
    seen = set()

    def abs_url(href: str) -> str:
        return urllib.parse.urljoin(tag_url, href) if href.startswith("/") else href

    EXCLUDES = (
        "/tags/", "/uebersicht", "/projektanmeldungen", "/termine", "/jury",
        "/genki", "/kuratiertes", "/cops", "/vernetzung",
        "/weiterbildung", "/wb-kompass", "/fobizz",
    )

    for a in links:
        href = a.get("href") or ""
        url = abs_url(href)
        if not url or url.endswith("#"):
            continue
        path = urllib.parse.urlparse(url).path.lower()

        # Nur echte Detailseiten:
        if "/home/innovationsfonds/projektvorstellungen/" not in path:
            continue
        if any(x in path for x in EXCLUDES):
            continue

        title = (a.get_text(" ", strip=True) or "").strip()
        if len(title) < 3:
            continue

        # Kontext: finde eine passende Karte/Container und hole p-Text
        ctx = a
        for parent in a.parents:
            if parent.name in ("article", "li", "div", "section"):
                ctx = parent
                break

        desc = ""
        p = ctx.find("p")
        if p and p.get_text(strip=True):
            desc = p.get_text(" ", strip=True)
        if not desc:
            sib = ctx.find_next_sibling()
            if sib:
                p2 = sib.find("p")
                if p2 and p2.get_text(strip=True):
                    desc = p2.get_text(" ", strip=True)
        if not desc:
            raw = ctx.get_text(" ", strip=True)
            desc = re.sub(r"\s+", " ", raw)

        # Dopplung entschärfen & kürzen
        if desc.lower().startswith(title.lower()):
            desc = desc[len(title):].lstrip(" :–-").strip()
        if len(desc) > 260:
            desc = desc[:257].rstrip() + "…"

        key = (url, title.lower())
        if key in seen:
            continue
        seen.add(key)

        items.append({"title": title, "url": url, "snippet": desc})

    # Dedupe nach URL & limit
    out, seen_u = [], set()
    for it in items:
        u = it.get("url")
        if not u or u in seen_u:
            continue
        seen_u.add(u)
        out.append(it)
        if len(out) >= max_items:
            break
    # Optional: Detail-Snippets für blasse Karten (max. 5 Requests)
    patched = 0
    for it in out:
        if it.get("snippet") and len(it["snippet"]) >= 60:
            continue
        u = it.get("url")
        if not u:
            continue
        detail = _fetch_detail_snippet(u, max_chars=400)
        if detail:
            it["snippet"] = detail
            patched += 1
        if patched >= 5:
            break
    return out

def _fetch_detail_snippet(url: str, max_chars: int = 400) -> str:
    """Holt einen kurzen Einleitungstext von der Projekt-Detailseite (best effort)."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(r.text, "html.parser")
    main = soup.select_one("main") or soup

    # Heuristik: erster sinnvoller Absatz
    for selector in ["article p", ".content p", "main p", "p"]:
        p = main.select_one(selector)
        if p and p.get_text(strip=True):
            txt = p.get_text(" ", strip=True)
            txt = re.sub(r"\s+", " ", txt).strip()
            if len(txt) > 40:  # sehr kurze Platzhalter vermeiden
                return (txt[:max_chars].rstrip() + "…") if len(txt) > max_chars else txt
    return ""
    
# ----------------------------------------------------------------------
# Datum tolerant aus Event holen (unterstützt: date_obj, date, ISO, dd.mm.yyyy)
# ----------------------------------------------------------------------
from datetime import date as _date
from typing import Optional

def _event_to_date(e) -> Optional[_date]:
    d = e.get("date_obj") or e.get("date")
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, _date):
        return d
    if isinstance(d, str):
        for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
            try:
                return datetime.strptime(d, fmt).date()
            except Exception:
                pass
    return None
# -----------------------------
# Live: Innovationsfonds per subject
# -----------------------------
def normalize_subject_to_slug(text: str) -> Optional[str]:
    """
    Verwendet SUBJECT_SLUGS und toleriert Umlaute in der Benutzerfrage.
    'Französisch' → 'franzoesisch' (Slug bleibt ASCII, exakt so gewünscht).
    """
    if not text:
        return None

    t = text.lower()
    # grobe Normalisierung (Umlaute & ß)
    t = (
        t.replace("ä", "ae")
         .replace("ö", "oe")
         .replace("ü", "ue")
         .replace("ß", "ss")
    )

    # zusätzlich beide Varianten prüfen (mit & ohne Umlaute)
    candidates = {t}
    candidates.add(text.lower())

    for cand in candidates:
        for key, slug in SUBJECT_SLUGS.items():
            if key in cand:
                return slug
    return None

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
SUBJECT_SLUGS = {
    "französisch": "franzoesisch",  # genau diese Schreibweise
    "franzoesisch": "franzoesisch",
    "englisch": "englisch",
    "deutsch": "deutsch",
    "chemie": "chemie",
    "mathematik": "mathematik",
    # ggf. ergänzen …
}

def normalize_subject(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    return SUBJECT_SLUGS.get(t)

def advanced_search(query: str, max_items: int = None):
    # ... your current ranking ...
    topk = max_items or RANK_TOPK
    return ranked[:topk]

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
def _as_dict(hit) -> Dict:
    """Akzeptiert Treffer als Dict oder (score, dict) und gibt immer ein Dict zurück."""
    if isinstance(hit, dict):
        return hit
    if isinstance(hit, tuple) and len(hit) >= 2 and isinstance(hit[1], dict):
        return hit[1]
    # Fallback – lieber leeres Dict als Exception im Produktivbetrieb
    return {}

def build_system_prompt() -> str:
    return (
        "Du bist der offizielle DLH Chatbot. Antworte auf Deutsch mit HTML-Formatierung. "
        "Wenn der Kontext mehrere Termin- oder Projektzeilen enthält, liste ALLE als eine HTML-Liste "
        "(<ul><li>…</li></ul>) auf. Nenne bei Terminen Datum und Zeit sowie einen verlinkten Titel. "
        "Bei Projekten im Innovationsfonds liste die Titel jeweils als klickbare Links. "
        "Wenn es sich um Termine/Workshops handelt, nutze folgendes HTML-Muster:\n\n"
        "<section class='dlh-answer'>\n"
        "  <p>Kurz-Einleitung (1 Satz).</p>\n"
        "  <ol class='timeline'>\n"
        "    <li>\n"
        "      <time>2025-11-11</time>\n"
        "      <a href='URL' target='_blank'>Titel des Workshops</a>\n"
        "      <div class='meta'>Ort/Format (falls bekannt)</div>\n"
        "    </li>\n"
        "    <!-- weitere <li> ... -->\n"
        "  </ol>\n"
        "  <h3>Quellen</h3>\n"
        "  <ul class='sources'>\n"
        "    <li><a href='URL' target='_blank'>Titel oder Domain</a></li>\n"
        "  </ul>\n"
        "</section>\n\n"
        "Bei Projektlisten nutze Karten:\n\n"
        "<section class='dlh-answer'>\n"
        "  <p>Kurz-Einleitung (1 Satz).</p>\n"
        "  <div class='cards'>\n"
        "    <article class='card'>\n"
        "      <h4><a href='URL' target='_blank'>Projekttitel</a></h4>\n"
        "      <p>Kurze Beschreibung (1–2 Sätze).</p>\n"
        "    </article>\n"
        "    <!-- weitere .card ... -->\n"
        "  </div>\n"
        "  <h3>Quellen</h3>\n"
        "  <ul class='sources'>\n"
        "    <li><a href='URL' target='_blank'>Titel oder Domain</a></li>\n"
        "  </ul>\n"
        "</section>\n"
    )
def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else (s[: max(0, n - 1)] + "…")    
def build_user_prompt(question: str, hits: List[Dict]) -> str:
    """Erzeugt den Prompt für das LLM mit Query, Datum und relevanten Textauszügen."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = [
        f"Heutiges Datum: {today}",
        f"Benutzerfrage: {question}",
        "",
        "Relevante Auszüge:"
    ]

    used = sum(len(p) for p in parts)  # aktueller Verbrauch
    for h in hits[:MAX_HITS_IN_PROMPT]:
        title = h.get("title") or h.get("metadata", {}).get("title") or "Ohne Titel"
        url   = h.get("url")   or h.get("metadata", {}).get("source") or ""
        snippet = (
            h.get("snippet")
            or h.get("content")
            or h.get("metadata", {}).get("description")
            or ""
        )
        block = f"- {title} — {url}\n  {_truncate(snippet, MAX_SNIPPET_CHARS)}\n"
        if used + len(block) > PROMPT_CHARS_BUDGET:
            break
        parts.append(block)
        used += len(block)


    parts.append(
        "Aufgabe: Antworte in sauberem HTML, wie im System-Prompt beschrieben. "
        "Verwende Listen (<ul><li>…</li></ul>) oder Karten, wenn sinnvoll. "
        "Verlinke Quellen mit <a href='URL' target='_blank'>Titel</a>."
    )

    return "\n".join(parts)
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
def build_sources(ranked, limit: int | None = None):
    lim = limit or MAX_SOURCES
    out = []
    seen = set()
    for h in ranked:
        url = h.get("metadata", {}).get("source") or h.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(SourceItem(
            title=h.get("metadata", {}).get("title") or h.get("title"),
            url=url,
            snippet=h.get("metadata", {}).get("snippet") or h.get("snippet")
        ))
        if len(out) >= lim:
            break
    return out
    
import html, re  # oben schon vorhanden? Wenn nicht, hier lassen.

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
# -----------------------------
# Endpoints
# -----------------------------
@app.get("/debug/deps")
def debug_deps():
    import platform
    import openai as _openai_pkg
    import bs4 as _bs4_pkg
    import lxml as _lxml_pkg
    return {
        "python": platform.python_version(),
        "openai": getattr(_openai_pkg, "__version__", "?"),
        "bs4": getattr(_bs4_pkg, "__version__", "?"),
        "lxml": getattr(_lxml_pkg, "__version__", "?"),
    }

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
    
# --- Live: Impuls-Workshops robust parsen ------------------------------------
IMPULS_URL = "https://dlh.zh.ch/home/impuls-workshops"

MONTHS = {
    "jan": 1, "januar": 1,
    "feb": 2, "februar": 2,
    "märz": 3, "maerz": 3, "mrz": 3,
    "apr": 4, "april": 4,
    "mai": 5,
    "jun": 6, "juni": 6,
    "jul": 7, "juli": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "okt": 10, "oktober": 10,
    "nov": 11, "november": 11,
    "dez": 12, "dezember": 12,
}

def _parse_date_de(s: str) -> Optional[datetime.date]:
    s0 = (s or "").strip().lower().replace("–", "-")
    # 1)  dd.mm.yyyy
    m = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(20\d{2})\b", s0)
    if m:
        d, mo, y = map(int, m.groups())
        try:
            return datetime(y, mo, d, tzinfo=timezone.utc).date()
        except ValueError:
            return None
    # 2)  dd mon yyyy  (z.B. "11 nov. 2025" / "11 nov 2025")
    m = re.search(r"\b(\d{1,2})\s*([a-zäöü\.]+)\s*(20\d{2})\b", s0)
    if m:
        d = int(m.group(1))
        mon_raw = m.group(2).strip(".")
        y = int(m.group(3))
        mo = MONTHS.get(mon_raw, MONTHS.get(mon_raw.replace("ä","ae").replace("ö","oe").replace("ü","ue")))
        if mo:
            try:
                return datetime(y, mo, d, tzinfo=timezone.utc).date()
            except ValueError:
                return None
    return None

def fetch_live_impuls_workshops() -> List[Dict]:
    """Gibt eine Liste von Events zurück: [{'date': date, 'title': str, 'url': str}]"""
    try:
        r = requests.get(IMPULS_URL, timeout=15, headers={"User-Agent": "DLH-Bot/1.0"})
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "lxml")

        # Cookie/Boilerplate verdrängen, falls stört (harmlos wenn leer)
        for sel in ["script", "style", "noscript", ".cookie", ".consent", ".banner"]:
            for el in soup.select(sel):
                el.decompose()

        root = soup.select_one("main") or soup

        events: List[Dict] = []

        # Variante A: Timeline/ol > li mit <time>
        for li in root.select("ol li, ul li"):
            txt = li.get_text(" ", strip=True)
            if not txt:
                continue
            date_el = li.find("time")
            date_str = (date_el.get_text(" ", strip=True) if date_el else txt)
            dt = _parse_date_de(date_str)
            a = li.find("a")
            title = a.get_text(" ", strip=True) if a else txt
            href = a.get("href") if a and a.has_attr("href") else ""
            if href and href.startswith("/"):
                href = urllib.parse.urljoin(IMPULS_URL, href)
            if dt and title:
                events.append({"date": dt, "title": title, "url": href or IMPULS_URL})

        # Variante B: Fallback – Abschnitt mit Überschrift „Termine der aktuellen…“
        if not events:
            h_candidates = [h for h in root.select("h2, h3") if "impuls" in h.get_text(" ", strip=True).lower()]
            for h in h_candidates:
                sec = h.find_next(["ol","ul","section","div"]) or root
                for li in sec.select("li"):
                    txt = li.get_text(" ", strip=True)
                    if not txt: 
                        continue
                    dt = _parse_date_de(txt)
                    a = li.find("a")
                    title = a.get_text(" ", strip=True) if a else txt
                    href = a.get("href") if a and a.has_attr("href") else ""
                    if href and href.startswith("/"):
                        href = urllib.parse.urljoin(IMPULS_URL, href)
                    if dt and title:
                        events.append({"date": dt, "title": title, "url": href or IMPULS_URL})

        # Deduplizieren (manchmal stehen Links doppelt)
        seen = set()
        cleaned = []
        for e in events:
            key = (e["date"].isoformat(), e["title"])
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(e)

        print(f"LIVE FETCH SUCCESS (Impuls): parsed {len(cleaned)} events (raw {len(events)})")
        return cleaned
    except Exception as ex:
        print("LIVE FETCH ERROR (Impuls):", repr(ex))
        return []
        
from datetime import datetime, timezone
import re

def _event_to_date(e: dict):
    """Versucht, aus einem Event-Dict ein date()-Objekt zu machen.
    Akzeptiert ISO, 'DD.MM.YYYY', 'D. Mon YYYY' usw. Liefert None bei Misserfolg."""
    if not e:
        return None
    # 1) Bereits dabei?
    d = e.get("date")
    if isinstance(d, datetime):
        return d.date()
    # 2) Aus Strings extrahieren
    for key in ("date", "date_str", "when", "datum"):
        val = e.get(key)
        if isinstance(val, str) and val.strip():
            s = val.strip()
            # ISO
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
            except Exception:
                pass
            # DD.MM.YYYY
            m = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(20\d{2})\b", s)
            if m:
                try:
                    return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1))).date()
                except Exception:
                    pass
            # 11. Nov 2025 / 11 Nov 2025
            m2 = re.search(r"\b(\d{1,2})\s*([A-Za-zÄÖÜäöü]+)\s*(20\d{2})\b", s)
            if m2:
                mois = {
                    "januar":"01","jan":"01","februar":"02","feb":"02","märz":"03","maerz":"03","mrz":"03",
                    "april":"04","apr":"04","mai":"05","juni":"06","jun":"06","juli":"07","jul":"07",
                    "august":"08","aug":"08","september":"09","sep":"09","oktober":"10","okt":"10",
                    "november":"11","nov":"11","dezember":"12","dez":"12"
                }
                mm = mois.get(m2.group(2).lower())
                if mm:
                    try:
                        return datetime(int(m2.group(3)), int(mm), int(m2.group(1))).date()
                    except Exception:
                        pass
    return None        
    
def render_workshops_html(items: List[Dict]) -> str:
    """Erzeugt eine kompakte HTML-Timeline mit klickbaren Titeln."""
    # Weiterleitung auf die neue Timeline-Version
    return render_workshops_timeline_html(items)

def render_workshops_timeline_html(events: list[dict], title: str = "Kommende Impuls-Workshops") -> str:
    """Erzeugt eine Timeline mit Datum + klickbaren Titeln (kompakt, valides HTML)."""
    if not events:
        return (
            "<section class='dlh-answer'>"
            "<p>Keine Workshops gefunden.</p>"
            "<h3>Quellen</h3>"
            "<ul class='sources'><li><a href='https://dlh.zh.ch/home/impuls-workshops' target='_blank'>Impuls-Workshop-Übersicht</a></li></ul>"
            "</section>"
        )

    lis = []
    for e in events:
        d = e.get("_d")  # hier liegt das bereits geparste date()
        date_str = d.strftime("%d.%m.%Y") if d else ""
        url = e.get("url") or e.get("link") or "https://dlh.zh.ch/home/impuls-workshops"
        t = e.get("title") or "Ohne Titel"
        lis.append(f"<li><time>{date_str}</time> <a href='{url}' target='_blank' rel='noopener'>{t}</a></li>")

    return (
        "<section class='dlh-answer'>"
        f"<p>{title}:</p>"
        "<ol class='timeline'>"
        + "".join(lis) +
        "</ol>"
        "<h3>Quellen</h3>"
        "<ul class='sources'><li><a href='https://dlh.zh.ch/home/impuls-workshops' target='_blank'>Impuls-Workshop-Übersicht</a></li></ul>"
        "</section>"
    )

    
def render_innovationsfonds_cards_html(items: List[Dict], subject_title: str, tag_url: str) -> str:
    if not items:
        return (
            "<section class='dlh-answer'>"
            f"<p>Für <strong>{subject_title}</strong> wurden aktuell keine Projektkarten gefunden.</p>"
            f"<p>Prüfe die <a href='{tag_url}' target='_blank'>Tag-Seite</a>.</p>"
            "</section>"
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
    
    # --- German month mapping + date parsing helpers ---
GER_MONTHS = {
    "jan": 1, "januar": 1,
    "feb": 2, "februar": 2,
    "mär": 3, "maerz": 3, "märz": 3,
    "apr": 4, "april": 4,
    "mai": 5,
    "jun": 6, "juni": 6,
    "jul": 7, "juli": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "okt": 10, "oktober": 10,
    "nov": 11, "november": 11,
    "dez": 12, "dezember": 12,
}

def _parse_german_date(text: str) -> Optional[str]:
    """
    Erkenne Datumsangaben wie '11 Nov. 2025' o.ä. und liefere 'YYYY-MM-DD'.
    """
    if not text:
        return None
    t = re.sub(r"\s+", " ", text, flags=re.I).strip()
    m = re.search(r"(\d{1,2})\s*\.?\s*([A-Za-zäöüÄÖÜ\.]+)\s+(\d{4})", t)
    if not m:
        return None
    day = int(m.group(1))
    mon_raw = m.group(2).lower().replace(".", "")
    mon_raw = mon_raw.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    month = GER_MONTHS.get(mon_raw)
    year = int(m.group(3))
    if not month or not (1 <= day <= 31):
        return None
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except Exception:
        return None  
        
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

# ----------------------------------------------------------------------
# Timeline-Renderer für Impuls-Workshops (neuer Stil)
# ----------------------------------------------------------------------
def render_workshops_timeline_html(events: list, title: str = "Impuls-Workshops") -> str:
    """Erzeugt eine strukturierte HTML-Timeline mit Datum, Titel und Link."""
    if not events:
        return f"<p>Keine Workshops gefunden.</p>"

    items_html = ""
    for e in events:
        date = e.get("date")
        if isinstance(date, datetime):
            date_str = date.strftime("%d.%m.%Y")
        else:
            date_str = str(date or "")
        title = e.get("title", "Ohne Titel")
        url = e.get("url", "#")
        desc = e.get("snippet") or e.get("summary") or ""
        items_html += f"""
        <li>
            <time>{date_str}</time>
            <a href="{url}" target="_blank">{title}</a>
            <div class="meta">{desc}</div>
        </li>
        """

    html = f"""
    <section class="dlh-answer">
      <p><strong>{title}</strong></p>
      <ol class="timeline">
        {items_html}
      </ol>
      <h3>Quellen</h3>
      <ul class="sources">
        <li><a href="https://dlh.zh.ch/home/impuls-workshops" target="_blank">
        Impuls-Workshop-Übersicht</a></li>
      </ul>
    </section>
    """
    return html

# ----------------------------------------------------------------------
# Timeline-Renderer für Impuls-Workshops
# ----------------------------------------------------------------------

def render_workshops_timeline_html(events: list[dict], title: str = "Kommende Impuls-Workshops") -> str:
    """Erzeugt eine Timeline mit Datum + klickbaren Titeln (kompakt, valides HTML)."""
    if not events:
        return (
            "<section class='dlh-answer'>"
            "<p>Keine Workshops gefunden.</p>"
            "<h3>Quellen</h3>"
            "<ul class='sources'><li><a href='https://dlh.zh.ch/home/impuls-workshops' target='_blank'>Impuls-Workshop-Übersicht</a></li></ul>"
            "</section>"
        )

    lis = []
    for e in events:
        d = e.get("_d")  # hier liegt das bereits geparste date()
        date_str = d.strftime("%d.%m.%Y") if d else ""
        url = e.get("url") or e.get("link") or "https://dlh.zh.ch/home/impuls-workshops"
        t = e.get("title") or "Ohne Titel"
        lis.append(f"<li><time>{date_str}</time> <a href='{url}' target='_blank' rel='noopener'>{t}</a></li>")

    return (
        "<section class='dlh-answer'>"
        f"<p>{title}:</p>"
        "<ol class='timeline'>"
        + "".join(lis) +
        "</ol>"
        "<h3>Quellen</h3>"
        "<ul class='sources'><li><a href='https://dlh.zh.ch/home/impuls-workshops' target='_blank'>Impuls-Workshop-Übersicht</a></li></ul>"
        "</section>"
    )

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    try:
        ranked = get_ranked_with_sitemap(req.question, max_items=req.max_sources or 12)
        print("Y  ranked types:", [type(x).__name__ for x in ranked[:5]])

        # Früher Exit für Workshop-Fragen: live scrapen & direkt rendern
        q_low = (req.question or "").lower()
        if any(k in q_low for k in ["impuls", "workshop", "workshops"]):
            # ---------- Workshops (Impuls) – Intent & Filter ----------
            def _norm(s: str) -> str:
                s = (s or "").lower()
                return (s.replace("ä","ae").replace("ö","oe").replace("ü","ue").replace("ß","ss"))

            qn = _norm(req.question)

            want_past = any(k in qn for k in [
                "gab es","waren","vergangenen","bisherigen","im jahr","letzten","fruehere","frühere","bisher"
            ])
            want_next = any(k in qn for k in [
                "naechste","nächste","der naechste","der nächste","als naechstes","als nächstes",
                "nur der naechste","nur der nächste","naechstes","nächstes"
            ])

            yr = None
            m = re.search(r"(?:jahr|jahrgang|seit)\s*(20\d{2})", qn)
            if m:
                try:
                    yr = int(m.group(1))
                except Exception:
                    yr = None

            raw = fetch_live_impuls_workshops() or []
            # → robuste Normalisierung inkl. _d = date()
            norm = []
            for e in raw:
                d = _event_to_date(e)
                if d:
                    ee = dict(e)
                    ee["_d"] = d
                    norm.append(ee)

            today = datetime.now(timezone.utc).date()
            future = [e for e in norm if e["_d"] >= today]
            past   = [e for e in norm if e["_d"] <  today]

            # Branching
            if want_next:
                events_to_show = sorted(future, key=lambda x: x["_d"])[:1]
                html = render_workshops_timeline_html(events_to_show, title="Nächster Impuls-Workshop")
                return AnswerResponse(
                    answer=html,
                    sources=[SourceItem(title="Impuls-Workshop-Übersicht", url="https://dlh.zh.ch/home/impuls-workshops")]
                )

            if want_past:
                if yr:
                    past = [e for e in past if e["_d"].year == yr]
                events_to_show = sorted(past, key=lambda x: x["_d"], reverse=True)
                html = render_workshops_timeline_html(events_to_show, title=f"Vergangene Impuls-Workshops{(' '+str(yr)) if yr else ''}")
                return AnswerResponse(
                    answer=html,
                    sources=[SourceItem(title="Impuls-Workshop-Übersicht", url="https://dlh.zh.ch/home/impuls-workshops")]
                )

            # Default: alle ab heute
            events_to_show = sorted(future, key=lambda x: x["_d"])
            html = render_workshops_timeline_html(events_to_show, title="Kommende Impuls-Workshops")
            return AnswerResponse(
                answer=html,
                sources=[SourceItem(title="Impuls-Workshop-Übersicht", url="https://dlh.zh.ch/home/impuls-workshops")]
            )
            # Default: alle ab heute
            future_sorted = sorted(future, key=lambda x: x["date"])
            html = render_workshops_timeline_html(
                future_sorted,
                title="Kommende Impuls-Workshops",
            )
            return AnswerResponse(
                answer=html,
                sources=[SourceItem(
                    title="Impuls-Workshop-Übersicht",
                    url=IMPULS_URL,
                )],
            )
            # Default: alle ab heute
            future_sorted = sorted(future, key=lambda x: x["date"])
            html = render_workshops_timeline_html(
                future_sorted,
                title="Kommende Impuls-Workshops",
            )
            return AnswerResponse(
                answer=html,
                sources=[
                    SourceItem(
                        title="Impuls-Workshop-Übersicht",
                        url="https://dlh.zh.ch/home/impuls-workshops",
                    )
                ],
            )

        # Früher Exit für Innovationsfonds-Projekte nach Fach (Cards)
        if any(k in q_low for k in ["innovationsfonds", "innovations-projekt", "innovationsprojekte", "projektvorstellungen"]):
            tag_slug = normalize_subject_to_slug(req.question)
            if tag_slug:
                tag_url = sitemap_find_innovations_tag(tag_slug)
                if tag_url:
                    cards = fetch_live_innovationsfonds_cards(tag_url)
                    if cards:
                        html = render_innovationsfonds_cards_html(
                            cards,
                            subject_title=tag_slug.capitalize(),
                            tag_url=tag_url,
                        )
                        srcs = [
                            SourceItem(
                                title=f"Innovationsfonds – {tag_slug}",
                                url=tag_url,
                                snippet=f"Projekte mit Tag {tag_slug}",
                            )
                        ]
                        # Optional: erste Projektseite als zweite Quelle
                        if cards and cards[0].get("url"):
                            srcs.append(
                                SourceItem(
                                    title=cards[0]["title"],
                                    url=cards[0]["url"],
                                    snippet=cards[0].get("snippet", ""),
                                )
                            )
                        return AnswerResponse(answer=html, sources=srcs)

        # Früher Exit für Innovationsfonds-Projekte nach Fach (Cards) – alternative Schreibweise
        if (
            "innovationsfonds" in q_low
            or "innovations-projekt" in q_low
            or "innovationsprojekte" in q_low
            or "projektvorstellungen" in q_low
            or "projekte" in q_low
            or "innovationsfond projekte" in q_low
            or "innovationsfonds projekte" in q_low
        ):
            tag_slug = normalize_subject_to_slug(req.question)
            if tag_slug:
                tag_url = sitemap_find_innovations_tag(tag_slug)
                if tag_url:
                    cards = fetch_live_innovationsfonds_cards(tag_url)
                    if cards:
                        html = render_innovationsfonds_cards_html(
                            cards,
                            subject_title=tag_slug.capitalize(),
                            tag_url=tag_url,
                        )
                        srcs = [
                            SourceItem(
                                title=f"Innovationsfonds – {tag_slug}",
                                url=tag_url,
                                snippet=f"Projekte mit Tag {tag_slug}",
                            )
                        ]
                        # optional: die erste Projekt-Detailseite als weitere Quelle
                        if cards and cards[0].get("url"):
                            srcs.append(
                                SourceItem(
                                    title=cards[0]["title"],
                                    url=cards[0]["url"],
                                    snippet=cards[0].get("snippet", ""),
                                )
                            )
                        return AnswerResponse(answer=html, sources=srcs)



        # LLM-Weg (wenn kein Workshop-/IF-Frühexit gegriffen hat)
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(req.question, ranked)

        print("Y  LLM call →", OPENAI_MODEL, "| prompt_len:", len(user_prompt))
        answer_html = call_openai(system_prompt, user_prompt, max_tokens=1200)
        answer_html = ensure_clickable_links(answer_html)

        sources = build_sources(ranked, limit=req.max_sources or 4)
        return AnswerResponse(answer=answer_html, sources=sources)

    except Exception as e:
        print("ERROR /ask:", repr(e))
        print(format_exc())
        msg = (
            "<strong>Entschuldigung, es gab einen technischen Fehler.</strong><br>"
            "Bitte versuchen Sie es später erneut."
        )
        return AnswerResponse(answer=msg, sources=[])


@app.get("/debug/validate")
def debug_validate():
    """Läuft ohne OpenAI-Call. Gut für schnelle Deploy-Checks."""
    res = validate_prompts()
    return res

# -----------------------------
# Local dev
# -----------------------------


# === Functions carried over from old live_patch2_fix (needed) ===
def build_system_prompt() -> str:
    return (
        "Du bist ein sachlicher Assistent des Digital Learning Hub Sek II (DLH Zürich). "
        "Sprich Deutsch. Antworte knapp, korrekt, ohne Spekulation. "
        "Formatiere die Antwort als valides HTML:\n"
        "- kurze Einleitung (1–2 Sätze)\n"
        "- strukturierte Liste (bullet points oder Timeline) der wichtigen Punkte\n"
        "- nutze <a href='URL' target='_blank'>Titel</a> für Links\n"
        "- schliesse mit einem Abschnitt <h3>Quellen</h3> und einer kurzen Liste der verwendeten URLs\n"
        "Wenn Informationen unsicher sind, kennzeichne dies und verlinke die Quelle."
    )
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
    
@app.on_event("startup")
def _run_prompt_validation():
    try:
        _v = validate_prompts()
        print("Y  prompt validation:", _v)
    except Exception as _e:
        print("Y  prompt validation ERROR:", repr(_e))



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



@app.on_event("startup")
def _sitemap_startup_loader():
    try:
        stats = load_sitemap(os.getenv("DLH_SITEMAP_PATH", "processed/dlh_sitemap.xml"))
        print("Y  sitemap loaded:", {"urls": stats.get("urls", 0), "sections": stats.get("sections", 0)})
    except Exception as e:
        print("Y  sitemap load ERROR:", repr(e))



@app.get("/debug/sitemap")
def debug_sitemap():
    return {
        "loaded": SITEMAP_LOADED,
        "urls": len(SITEMAP_URLS),
        "sections": {k: len(v) for k, v in SITEMAP_SECTIONS.items()}
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

def sitemap_find_innovations_tag(tag_slug: str) -> Optional[str]:
    """
    Liefert die Tag-Seite für den Innovationsfonds aus der Sitemap.
    Fallback: konstruiere die bekannte Tag-URL direkt.
    """
    if not tag_slug:
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
