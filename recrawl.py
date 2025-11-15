import json
import re
import urllib.parse
import logging
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional
from pathlib import Path
from bs4 import BeautifulSoup, Tag

# --- Konfiguration ---
# Der Ordner, in dem Ihre Offline-HTML-Dateien liegen
BASE_DIR = Path("processed/dlh-zh-ch") 
# Die Zieldatei, die von 'build_database.py' gelesen wird
OUTPUT_FILE = Path("processed/processed_chunks.json") 
BASE_URL = "https://dlh.zh.ch"
# --- Ende Konfiguration ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Die vom Benutzer bereitgestellte Subject Map
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

def clean_html_content(soup: BeautifulSoup) -> BeautifulSoup:
    """Entfernt alle unerwünschten Tags aus dem Soup-Objekt."""
    junk_tags = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'svg', 'img', 'noscript', 'meta', 'link']
    for tag_name in junk_tags:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    
    junk_selectors = [
        ".cookie-banner", ".breadcrumb", ".user-menu", ".search-form", 
        "#header", "#footer", ".nav", ".navigation", ".pagination",
        ".social-links", ".skip-links"
    ]
    for selector in junk_selectors:
        for tag in soup.select(selector):
            tag.decompose()
    
    return soup

def reconstruct_url(file_path: Path, base_dir: Path, item_href: Optional[str] = None) -> str:
    """
    Erstellt die Live-URL aus dem Dateipfad oder einem href-Link.
    """
    if item_href:
        if item_href.startswith(('http://', 'https://')):
            return item_href 
        if item_href.startswith('#'):
             pass 
        elif item_href.startswith('/'):
            return f"{BASE_URL}{item_href}"
        else:
            base_url_path = reconstruct_url(file_path, base_dir)
            return urllib.parse.urljoin(base_url_path, item_href)

    relative_path = file_path.relative_to(base_dir)
    url_path = str(relative_path.as_posix())
    
    if url_path.endswith("index.html"):
        url_path = url_path[:-10] 
    elif url_path.endswith(".html"):
        url_path = url_path[:-5] 
    
    if url_path and not url_path.startswith('/'):
            url_path = '/' + url_path
    
    if not Path(file_path).suffix and not url_path.endswith('/'):
        url_path += '/'
        
    return f"{BASE_URL}{url_path}"

def extract_metadata_rules(soup_or_tag: BeautifulSoup, url: str, text_content: str) -> Dict:
    """Wendet spezifische Regeln an, um strukturierte Metadaten zu finden."""
    metadata = {}
    
    # Regel 1: Innovationsfonds-Projekte
    if "innovationsfonds/projektvorstellungen/uebersicht" in url and "filterergebnisse" not in url:
        metadata['chunk_type'] = 'innovationsfonds_project'
    
    # Regel 2: Fobizz
    if "fobizz" in url or "fobizz" in text_content.lower():
        metadata['chunk_type'] = 'fobizz_resource'

    # Regel 3: Workshops (Datums-Extraktion)
    dates_found = set()
    DMY_DOTTED_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b")
    DMY_TEXT_RE = re.compile(r"\b(\d{1,2})\.\s*([A-Za-zäöüÄÖÜ]+)\s*(\d{2,4})?\b", re.IGNORECASE)
    
    for match in DMY_DOTTED_RE.finditer(text_content):
        dates_found.add(match.group(0))
    for match in DMY_TEXT_RE.finditer(text_content):
        dates_found.add(match.group(0)) 
    
    if dates_found:
        metadata['dates'] = list(dates_found)

    return metadata

def parse_page_to_chunks(soup: BeautifulSoup, page_url: str, page_title: str) -> List[Dict]:
    """Extrahiert Chunks (entweder als Liste oder semantische Blöcke) aus einer Soup."""
    chunks = []
    
    # --- STRATEGIE A: Listen-Extraktion (Smart Chunking) ---
    item_selectors = [
        "article.card",          # Standard-Karten (z.B. Innovationsfonds)
        ".item-list .item",      # Joomla/Allgemeine Listen
        ".blog-item",            # Blog-Listen
        "article.uk-article"     # UI-Kit Artikel (oft für Projekte/News)
    ]
    
    list_items = soup.select(", ".join(item_selectors))
    
    if list_items and len(list_items) > 2:
        logging.info(f"[List Page] {page_url} -> {len(list_items)} items found.")
        
        for item_soup in list_items:
            item_text = item_soup.get_text(" ", strip=True)
            
            if len(item_text.split()) < 5: 
                continue
                
            title_tag = item_soup.find(['h2', 'h3', 'h4'])
            item_title = title_tag.get_text(strip=True) if title_tag else page_title
            
            link_tag = item_soup.find('a', href=True)
            item_url = reconstruct_url(Path.cwd(), BASE_DIR, link_tag['href']) if (link_tag and link_tag.get('href')) else page_url

            metadata = extract_metadata_rules(item_soup, item_url, item_text)
            metadata['source'] = item_url
            metadata['title'] = item_title
            metadata['crawled_at'] = datetime.now(timezone.utc).isoformat()
            metadata['chunk_type'] = metadata.get('chunk_type', 'list_item') 

            chunks.append({
                "content": item_text,
                "url": item_url, 
                "title": item_title, 
                "metadata": metadata
            })
        
        return chunks
    
    # --- STRATEGIE B: Fallback (Seiten-Modus / Semantische Blöcke) ---
    main_content = soup.find('main') or soup.find('body') or soup
    if not main_content:
        return []
            
    # 1. Chunk (Titel/H1)
    h1_tag = main_content.find('h1')
    h1_text = h1_tag.get_text(strip=True) if h1_tag else page_title
    
    if h1_text:
         metadata = extract_metadata_rules(main_content, page_url, h1_text)
         metadata['source'] = page_url
         metadata['title'] = h1_text
         metadata['crawled_at'] = datetime.now(timezone.utc).isoformat()
         metadata['chunk_type'] = metadata.get('chunk_type', 'title')
         chunks.append({
            "content": h1_text,
            "url": page_url, 
            "title": page_title, 
            "metadata": metadata
         })

    # 2. Chunks (Absätze und Listen)
    semantic_tags = ['p', 'li', 'td']
    
    for tag in main_content.find_all(semantic_tags):
        text_content = tag.get_text(" ", strip=True)
        
        if not text_content or len(text_content.split()) < 5:
            continue
            
        if tag.name == 'li' and tag.find('a') and len(tag.find_all(text=True)) < 10:
            continue 

        metadata = extract_metadata_rules(main_content, page_url, text_content)
        metadata['source'] = page_url
        metadata['title'] = page_title
        metadata['crawled_at'] = datetime.now(timezone.utc).isoformat()
        metadata['chunk_type'] = tag.name 

        chunks.append({
            "content": text_content,
            "url": page_url, 
            "title": page_title, 
            "metadata": metadata
        })
        
    return chunks

def process_offline_file(file_path: Path, base_dir: Path) -> List[Dict]:
    """Liest, parst und extrahiert Daten aus einer einzelnen OFFLINE-HTML-Datei."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        page_url = reconstruct_url(file_path, base_dir)
        page_title = (soup.title.string or "Ohne Titel").strip()
        page_title = page_title.split("– Digital Learning Hub")[0].strip()

        cleaned_soup = clean_html_content(soup)
        
        return parse_page_to_chunks(cleaned_soup, page_url, page_title)

    except Exception as e:
        logging.error(f"Fehler beim Verarbeiten von {file_path}: {e}")
        return []

def fetch_online_projects(subject_map: Dict) -> List[Dict]:
    """
    NEU: Crawlt die Innovationsfonds-Filterseiten ONLINE, um alle Projekt-URLs zu finden
    und crawlt dann jede Projektseite.
    """
    logging.info("Starte ONLINE-Crawl für Innovationsfonds-Projekte...")
    
    base_filter_url = "https://dlh.zh.ch/home/innovationsfonds/projektvorstellungen/uebersicht/filterergebnisse-fuer-projekte/tags/"
    project_urls_to_crawl = set()
    
    headers = {"User-Agent": "DLH-Bot-Crawler/1.0 (Python)"}

    # 1. Finde alle Projekt-URLs, indem die Filterseiten gecrawlt werden
    for subject_key in subject_map.keys():
        filter_url = base_filter_url + subject_key
        logging.info(f"Crawle Filter-URL: {filter_url}")
        try:
            response = requests.get(filter_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Finde alle Links, die wie Projekt-Detailseiten aussehen
            # (z.B. /uebersicht/1078-baustelle-berufsfachschule)
            project_links = soup.select('a[href*="/uebersicht/"]')
            
            for link in project_links:
                href = link['href']
                # Wir suchen nach dem spezifischen Muster: /Zahl-Text
                if re.search(r'/\d+-[a-z0-9-]+', href):
                    full_url = urllib.parse.urljoin(BASE_URL, href)
                    project_urls_to_crawl.add(full_url)
                    
        except Exception as e:
            logging.warning(f"Konnte Filter-Seite nicht crawlen: {filter_url} | Fehler: {e}")
    
    logging.info(f"{len(project_urls_to_crawl)} einzigartige Projekt-URLs gefunden. Starte Detail-Crawl...")

    # 2. Crawle jede einzelne Projekt-URL für den Inhalt
    online_chunks = []
    for project_url in project_urls_to_crawl:
        try:
            response = requests.get(project_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            page_title = (soup.title.string or "Ohne Titel").strip()
            page_title = page_title.split("– Digital Learning Hub")[0].strip()

            cleaned_soup = clean_html_content(soup)
            
            # Wir behandeln diese Seite als Detailseite (Strategie B)
            project_chunks = parse_page_to_chunks(cleaned_soup, project_url, page_title)
            
            # Wichtig: Markiere alle Chunks dieser Seite explizit als 'innovationsfonds_project'
            for ch in project_chunks:
                ch['metadata']['chunk_type'] = 'innovationsfonds_project'
                ch['metadata']['fach'] = [subject for key, subject in subject_map.items() if f"/tags/{key}" in response.url] # Versuch, das Fach zuzuordnen
            
            online_chunks.extend(project_chunks)
            logging.info(f"-> {len(project_chunks)} Chunks für {project_url} extrahiert.")
            
        except Exception as e:
            logging.warning(f"Konnte Projekt-Seite nicht crawlen: {project_url} | Fehler: {e}")
            
    return online_chunks


def main_crawler():
    """Haupt-Crawler-Funktion."""
    if not BASE_DIR.is_dir():
        logging.error(f"FATAL: Basisverzeichnis nicht gefunden: {BASE_DIR.resolve()}")
        logging.error("Bitte stellen Sie sicher, dass der Ordner 'processed/dlh-zh-ch' existiert.")
        return

    all_chunks = []
    
    # --- SCHRITT 1: OFFLINE CRAWL (Fobizz, Workshops, etc.) ---
    html_files = list(BASE_DIR.rglob("*.html")) + list(BASE_DIR.rglob("*.htm"))
    
    if not html_files:
        logging.error(f"Keine .html-Dateien in {BASE_DIR.resolve()} gefunden.")
    else:
        logging.info(f"Starte OFFLINE Crawl von {len(html_files)} Dateien in {BASE_DIR}...")
        for file_path in html_files:
            chunks = process_offline_file(file_path, BASE_DIR) 
            if chunks:
                all_chunks.extend(chunks)
    
    logging.info(f"{len(all_chunks)} Chunks aus Offline-Dateien extrahiert.")

    # --- SCHRITT 2: ONLINE CRAWL (Innovationsfonds) ---
    online_project_chunks = fetch_online_projects(SUBJECT_MAP)
    all_chunks.extend(online_project_chunks)

    # --- SCHRITT 3: Speichern ---
    try:
        # Sicherstellen, dass der processed-Ordner existiert
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            
        logging.info(f"✅✅✅ ERFOLG! {len(all_chunks)} GESAMT-Chunks wurden in {OUTPUT_FILE} gespeichert.")
        logging.info(f"NÄCHSTER SCHRITT: Führen Sie 'python build_database.py' aus, um diese Chunks zu verarbeiten.")

    except Exception as e:
        logging.error(f"Fehler beim Speichern der JSON-Datei: {e}")

if __name__ == "__main__":
    main_crawler()