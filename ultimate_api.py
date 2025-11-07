
import os
import json
import re
import urllib.parse
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# OpenAI client (Chat Completions)
# -----------------------------
from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID") or None,
)

def call_openai(system_prompt: str, user_prompt: str, max_tokens: int = 1200, temperature: float = 0.2) -> str:
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# -----------------------------
# App / Models
# -----------------------------
app = FastAPI(title="DLH Chatbot API (OpenAI)")

class QuestionRequest(BaseModel):
    question: str
    language: Optional[str] = "de"
    max_sources: Optional[int] = 8

class AnswerResponse(BaseModel):
    answer: str


# -----------------------------
# Load processed chunks (cache)
# -----------------------------
CHUNKS: List[Dict] = []
ROOT = Path(__file__).parent
CANDIDATE_PATHS = [
    ROOT / "processed" / "processed_chunks.json",
    ROOT / "processed_chunks.json",
    Path("/mnt/data/processed_chunks.json"),
]

def load_chunks() -> None:
    global CHUNKS
    for p in CANDIDATE_PATHS:
        if p.exists():
            try:
                CHUNKS = json.loads(p.read_text(encoding="utf-8"))
                print(f"Loaded {len(CHUNKS)} chunks from {p}")
                return
            except Exception as e:
                print("Failed to load chunks:", e)
    CHUNKS = []
    print("No processed_chunks.json found; proceeding without cache.")

load_chunks()


# -----------------------------
# Helpers: German date parsing
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
    return s.replace("\u2013", "-").replace("\u2014", "-").replace("–", "-").replace("—", "-")

def parse_de_date_text(txt: str) -> Optional[datetime]:
    """Parse e.g. '11. Nov 2025, 17:15 Uhr – 18:00 Uhr' into datetime (start)."""
    if not txt:
        return None
    s = _normalize_dash(txt.lower().strip())
    s = s.replace("uhr", "").replace(",", " ")
    # spelled month
    m = re.search(r"(\\d{1,2})\\s*(?:\\.\\s*)?(jan|januar|feb|februar|maerz|märz|mrz|mar|april|apr|mai|jun|juni|jul|juli|aug|august|sep|sept|september|okt|oktober|nov|november|dez|dezember)\\s*(\\d{4})", s)
    if m:
        day = int(m.group(1))
        mon = m.group(2)
        month = DE_MONTHS.get(mon, None)
        year = int(m.group(3))
    else:
        # dd.mm.yyyy
        m2 = re.search(r"(\\d{1,2})\\.(\\d{1,2})\\.(\\d{4})", s)
        if not m2:
            return None
        day, month, year = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
    # time
    tmatch = re.search(r"(\\d{1,2}):(\\d{2})", s)
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
# Live: Innovationsfonds (Tag-Seiten je Fach)
# -----------------------------
SUBJECT_SLUGS = {
    "chemie": "chemie",
    "physik": "physik",
    "biologie": "biologie",
    "mathematik": "mathematik",
    "informatik": "informatik",
    "deutsch": "deutsch",
    "englisch": "englisch",
    "franzoesisch": "franzoesisch",  # (nur diese Schreibweise)
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
# Intent & Search
# -----------------------------
def normalize_query(q: str) -> Tuple[str, str]:
    ql = q.lower()
    # Nur für Facherkennung Umlaute normalisieren
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
        "franzoesisch": ["franzoesisch"],  # nur diese Schreibweise
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

    # 1) Forciere Live-Fetch für Workshops/Termine
    if intent["is_date_query"] or any(k in query_lower for k in ["impuls", "workshop", "termine", "veranstaltung", "events"]):
        live_chunk = fetch_live_impuls_workshops()
        if live_chunk:
            results.append((220, live_chunk))

    # 2) Forciere Live-Fetch für Innovationsfonds bei Facherkennung (auch ohne explizites Wort 'Innovationsfonds')
    if intent["subject_keywords"]:
        for subj in intent["subject_keywords"]:
            live_proj = fetch_live_innovationsfonds(subject=subj)
            if live_proj:
                results.append((300, live_proj))

    # 3) Fallback: gecrawlte Chunks (niedrige Priorität)
    for ch in CHUNKS[:256]:  # begrenzt
        results.append((120, ch))

    # nach Score sortieren (höchster zuerst)
    results.sort(key=lambda x: x[0], reverse=True)
    # deduplizieren per source+title
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
        # truncate very long content
        if len(content) > 3500:
            content = content[:3500] + "…"
        lines.append(f"[{score}] {title} <{src}>")
        lines.append(content)
        lines.append("")
    lines.append("AUFGABE: Antworte praezise und knapp. Bei Termin- oder Projektlisten verwende HTML-Listen.")
    return "\n".join(lines)


# -----------------------------
# FastAPI endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(CHUNKS), "model": OPENAI_MODEL}

@app.get("/version")
def version():
    return {"version": "openai-backend", "model": OPENAI_MODEL}

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    ranked = advanced_search(req.question, max_items=req.max_sources or 8)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(req.question, ranked)
    answer = call_openai(system_prompt, user_prompt, max_tokens=1200, temperature=0.2)
    return AnswerResponse(answer=answer)


# -----------------------------
# For local run: uvicorn ultimate_api:app --host 0.0.0.0 --port 8000
# -----------------------------
