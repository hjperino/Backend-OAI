import json
import re
from collections import defaultdict
from pathlib import Path
import logging
from datetime import datetime, timezone, date
from typing import List, Dict, Optional, Set

# --- HILFSFUNKTIONEN FÜR DATUMS-PARSING ---
# (Kopiert aus ultimate_api.py, notwendig für die Workshop-Extraktion)

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

def coerce_year(y, refyear):
    if not y: return refyear
    y = y.strip()
    if len(y) == 2: y = "20" + y
    try: return int(y)
    except Exception: return refyear

def parse_de_date(text: str, ref_date: Optional[datetime] = None) -> Optional[datetime]:
    """Versucht ein Datum (und ggf. Uhrzeit) aus deutschem Text zu extrahieren."""
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

# --- DATENBANK-LOGIK ---

def load_chunks(path: str) -> List[Dict]:
    """Lädt die Roh-Chunks."""
    p = Path(path)
    if not p.exists():
        print(f"Fehler: {path} nicht gefunden.")
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Fehler beim Laden der Chunks: {e}")
        return []

def extract_innovationsfonds_projects(chunks: List[Dict]) -> List[Dict]:
    """Extrahiert und bereinigt Innovationsfonds-Projekte."""
    projects = {} # Verwende ein Dict, um Chunks nach URL zu gruppieren
    
    for ch in chunks:
        # KORRIGIERTE REGEL: Suche nach dem von recrawl.py gesetzten Metadaten-Typ
        if ch.get('metadata', {}).get('chunk_type') == 'innovationsfonds_project':
            url = ch.get('url')
            if not url:
                continue

            # --- DAS IST DIE KORREKTUR ---
            # Wir ignorieren NUR die exakte Übersichtsseite, nicht die Detailseiten
            if url.endswith("/innovationsfonds/projektvorstellungen/uebersicht") or url.endswith("/innovationsfonds/projektvorstellungen/uebersicht/"):
                continue
            # --- ENDE DER KORREKTUR ---

            if url not in projects:
                # Initialisiere das Projekt mit dem Titel-Chunk
                projects[url] = {
                    "title": ch.get('metadata', {}).get('title', ch.get('title', 'Projekt ohne Titel')),
                    "url": url,
                    "description": ch.get('content', '') # Beginne mit dem ersten Chunk-Inhalt
                }
            else:
                # Füge weitere Inhalte (Absätze) hinzu, wenn der Chunk-Typ passt
                if ch.get('metadata', {}).get('chunk_type') != 'title':
                    projects[url]["description"] += "\n" + ch.get('content', '')

    # Konvertiere das Dict zurück in eine Liste
    return list(projects.values())

def extract_fobizz_resources(chunks: List[Dict]) -> List[Dict]:
    """Extrahiert Fobizz-Angebote."""
    items = {} # Dict zur Deduplizierung nach URL
    for ch in chunks:
        if ch.get('metadata', {}).get('chunk_type') == 'fobizz_resource':
            url = ch.get('url')
            if not url:
                continue
                
            if url not in items:
                items[url] = {
                    "title": ch.get("title", "Fobizz Angebot"),
                    "url": url,
                    "description": ch.get("content", "")
                }
            else:
                if ch.get('metadata', {}).get('chunk_type') != 'title':
                    items[url]["description"] += "\n" + ch.get("content", "")

    return list(items.values())

def extract_workshops(chunks: List[Dict]) -> List[Dict]:
    """Extrahiert alle Workshops und Termine aus den Chunks."""
    events = []
    seen_keys = set()
    for ch in chunks:
        # Regel: Muss "impuls-workshops" ODER Metadaten['dates'] haben
        is_event_page = "impuls-workshops" in ch.get('url', '').lower()
        dates_in_meta = ch.get('metadata', {}).get('dates', [])
        
        if is_event_page or dates_in_meta:
            
            for d_str in dates_in_meta:
                dt_obj = parse_de_date(d_str) 
                if dt_obj:
                    title = ch.get("title", "Termin")
                    # Reinige den Titel, falls er das Datum enthält (wie von recrawl.py hinzugefügt)
                    title = title.replace(f" ({d_str})", "") 
                    
                    url = ch.get('url', '#')
                    key = (title, url, dt_obj.isoformat())
                    
                    if key not in seen_keys:
                        events.append({
                            "title": title,
                            "url": url,
                            "date_iso": dt_obj.isoformat(), # Wichtig für die Sortierung
                            "date_str": d_str
                        })
                        seen_keys.add(key)
                        
    # Sortieren der Events
    return sorted(events, key=lambda x: x['date_iso'])


def main():
    print("Starte Offline-Datenbank-Erstellung...")
    
    # 1. Laden
    raw_chunks = load_chunks("processed/processed_chunks.json")
    if not raw_chunks:
        return
        
    # 2. Extrahieren
    projects = extract_innovationsfonds_projects(raw_chunks)
    fobizz = extract_fobizz_resources(raw_chunks)
    workshops = extract_workshops(raw_chunks)
    
    # 3. Datenbank-Objekt erstellen
    database = {
        "innovationsfonds_projects": projects,
        "fobizz_resources": fobizz,
        "workshops_kb": workshops,
    }
    
    # 4. Speichern
    output_path = "processed/structured_db.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) # Stelle sicher, dass der Ordner existiert
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Erfolg! {len(projects)} Projekte, {len(fobizz)} Fobizz-Einträge und {len(workshops)} Workshops gespeichert in {output_path}")

if __name__ == "__main__":
    main()