"""
Ultimate API server für DLH Chatbot - FINAL OPTIMIZED VERSION
Features:
- Correct file path: processed/processed_chunks.json
- Searches in metadata "fächer" field for subject-specific projects
- Guaranteed links for Innovationsfonds projects
- Enhanced date extraction for events
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import json
import os
import re
import uvicorn
from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import Counter

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="DLH Chatbot API (System Prompt Version)",
    description="AI-powered chatbot für dlh.zh.ch - mit System Prompt für garantierte Links",
    version="3.6.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic client
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Load and preprocess data
def load_and_preprocess_data():
    """Lade und bereite Daten mit verbesserter Struktur vor"""
    try:
        # Correct path based on GitHub structure
        file_path = 'processed/processed_chunks.json'
        
        print(f"🔍 Attempting to load data from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"✅ Successfully loaded {len(chunks)} chunks from {file_path}")
        
        # Erstelle Index für schnellere Suche
        keyword_index = {}
        url_index = {}
        subject_index = {}  # NEU: Index für Fächer
        
        for i, chunk in enumerate(chunks):
            # URL-basierter Index
            url = chunk['metadata'].get('source', '').lower()
            if url not in url_index:
                url_index[url] = []
            url_index[url].append(i)
            
            # NEU: Fächer-Index aus Metadaten
            faecher = chunk['metadata'].get('fächer', [])
            if faecher:
                for fach in faecher:
                    fach_lower = fach.lower()
                    if fach_lower not in subject_index:
                        subject_index[fach_lower] = []
                    subject_index[fach_lower].append(i)
            
            # Keyword-Index für wichtige Begriffe
            content = chunk['content'].lower()
            important_terms = [
                'fobizz', 'genki', 'innovationsfonds', 'cop', 'cops',
                'vernetzung', 'workshop', 'weiterbildung', 'kuratiert',
                'impuls', 'termin', 'anmeldung', 'lunch', 'learn',
                'impuls-workshop', 'impulsworkshop', 'veranstaltung', 'event',
                # Fächer (auch im Content suchen)
                'chemie', 'physik', 'biologie', 'mathematik', 'informatik',
                'deutsch', 'englisch', 'französisch', 'italienisch', 'spanisch',
                'geschichte', 'geografie', 'wirtschaft', 'recht', 'philosophie',
                'psychologie', 'pädagogik', 'kunst', 'musik', 'sport'
            ]
            
            for term in important_terms:
                if term in content:
                    if term not in keyword_index:
                        keyword_index[term] = []
                    keyword_index[term].append(i)
        
        print(f"🔍 Indexed {len(keyword_index)} keywords")
        print(f"📚 Indexed {len(subject_index)} subjects in metadata")
        if subject_index:
            print(f"   Subjects found: {', '.join(subject_index.keys())}")
        
        return chunks, keyword_index, url_index, subject_index
    except FileNotFoundError as e:
        print(f"❌ ERROR: File not found: {e}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Files in current directory: {os.listdir('.')}")
        return [], {}, {}, {}
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return [], {}, {}, {}

# Global data storage
CHUNKS, KEYWORD_INDEX, URL_INDEX, SUBJECT_INDEX = load_and_preprocess_data()

class QuestionRequest(BaseModel):
    question: str
    language: Optional[str] = "de"
    max_sources: Optional[int] = 5

class Source(BaseModel):
    url: str
    title: str
    snippet: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]

def extract_dates_from_text(text: str) -> List[Tuple[datetime, str]]:
    """
    Extrahiere Daten aus Text - unterstützt auch abgekürzte Monatsnamen
    """
    dates_found = []
    
    month_map_full = {
        'januar': 1, 'februar': 2, 'märz': 3, 'april': 4,
        'mai': 5, 'juni': 6, 'juli': 7, 'august': 8,
        'september': 9, 'oktober': 10, 'november': 11, 'dezember': 12
    }
    
    month_map_abbr = {
        'jan': 1, 'feb': 2, 'mär': 3, 'märz': 3, 'mrz': 3, 'apr': 4,
        'mai': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'sept': 9, 'okt': 10, 'nov': 11, 'dez': 12
    }
    
    patterns = [
        (r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})', 'numeric'),
        (r'(\d{1,2})\.\s*(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s*(\d{4})', 'full_month'),
        (r'(\d{1,2})\.?\s+(Jan\.?|Feb\.?|Mär\.?|März\.?|Mrz\.?|Apr\.?|Mai\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Okt\.?|Nov\.?|Dez\.?)\s+(\d{4})', 'abbr_month'),
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

def sort_events_chronologically(chunks: List[Dict], current_date: datetime = None) -> Dict[str, List[Dict]]:
    """
    Sortiere Events chronologisch und trenne vergangene von zukünftigen Events
    """
    if current_date is None:
        current_date = datetime.now()
    
    future_events = []
    past_events = []
    no_date_events = []
    
    for chunk in chunks:
        content = chunk['content']
        dates = extract_dates_from_text(content)
        
        if dates:
            dates.sort(key=lambda x: x[0])
            earliest_date = dates[0][0]
            
            event_info = {
                'chunk': chunk,
                'date': earliest_date,
                'date_str': dates[0][2],
                'all_dates': dates,
                'context': dates[0][1]
            }
            
            if earliest_date.date() < current_date.date():
                past_events.append(event_info)
            else:
                future_events.append(event_info)
        else:
            no_date_events.append({'chunk': chunk})
    
    future_events.sort(key=lambda x: x['date'])
    past_events.sort(key=lambda x: x['date'], reverse=True)
    
    return {
        'future_events': future_events,
        'past_events': past_events,
        'no_date_events': no_date_events
    }

def extract_query_intent(query: str) -> Dict[str, any]:
    """Analysiere die Absicht der Frage mit verbesserter Fach-Erkennung"""
    query_lower = query.lower()
    
    # Innovationsfonds/Projekt-Begriffe
    innovationsfonds_terms = [
        'innovationsfonds', 'innovationsprojekt', 'innovationsprojekte',
        'projekt', 'projekte', 'welche projekte'
    ]
    
    intent = {
        'is_date_query': any(term in query_lower for term in ['heute', 'morgen', 'termin', 'wann', 'datum', 'zeit', 'event', 'veranstaltung', 'nächste', 'kommende']),
        'is_how_to': any(term in query_lower for term in ['wie', 'anleitung', 'tutorial', 'schritte']),
        'is_definition': any(term in query_lower for term in ['was ist', 'was sind', 'definition', 'bedeutung']),
        'wants_list': any(term in query_lower for term in ['welche', 'liste', 'alle', 'überblick', 'übersicht']),
        'wants_contact': any(term in query_lower for term in ['kontakt', 'anmeldung', 'email', 'telefon', 'anmelden']),
        'is_innovationsfonds_query': any(term in query_lower for term in innovationsfonds_terms),
        'topic_keywords': [],
        'subject_keywords': []  # NEU: Speziell für Fächer
    }
    
    # Themenerkennung
    topics = {
        'fobizz': ['fobizz', 'to teach', 'to-teach'],
        'genki': ['genki', 'gen ki', 'gen-ki'],
        'innovationsfonds': ['innovationsfonds', 'innovation', 'projekt'],
        'workshop': ['workshop', 'impuls', 'veranstaltung', 'impuls-workshop', 'impulsworkshop', 'event'],
        'cop': ['cop', 'cops', 'community', 'practice'],
        'weiterbildung': ['weiterbildung', 'fortbildung', 'kurs', 'schulung'],
        'vernetzung': ['vernetzung', 'netzwerk', 'austausch'],
        'kuratiert': ['kuratiert', 'kuratiertes', 'sammlung']
    }
    
    # NEU: Fächer separat behandeln
    subjects = {
        'chemie': ['chemie'],
        'physik': ['physik'],
        'biologie': ['biologie'],
        'mathematik': ['mathematik', 'mathe'],
        'informatik': ['informatik'],
        'deutsch': ['deutsch'],
        'englisch': ['englisch'],
        'französisch': ['französisch'],
        'italienisch': ['italienisch'],
        'spanisch': ['spanisch'],
        'geschichte': ['geschichte'],
        'geografie': ['geografie', 'geographie'],
        'wirtschaft': ['wirtschaft'],
        'recht': ['recht'],
        'philosophie': ['philosophie'],
        'psychologie': ['psychologie'],
        'pädagogik': ['pädagogik'],
        'kunst': ['kunst'],
        'musik': ['musik'],
        'sport': ['sport']
    }
    
    for topic, keywords in topics.items():
        if any(kw in query_lower for kw in keywords):
            intent['topic_keywords'].append(topic)
    
    # NEU: Fächer separat erkennen
    for subject, keywords in subjects.items():
        if any(kw in query_lower for kw in keywords):
            intent['subject_keywords'].append(subject)
    
    return intent

def advanced_search(query: str, max_results: int = 10) -> List[Dict]:
    """ENHANCED: Suche mit Priorisierung von Metadaten-Fächern"""
    intent = extract_query_intent(query)
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    results = []
    
    # Spezialbehandlung für Impuls-Workshops
    if 'impuls' in query_lower and 'workshop' in query_lower:
        intent['topic_keywords'].append('impulsworkshop')
    
    # NEU: Spezielle Behandlung für Fach-spezifische Innovationsfonds-Anfragen
    if intent['is_innovationsfonds_query'] and intent['subject_keywords']:
        print(f"🔍 Searching for Innovationsfonds projects in subjects: {intent['subject_keywords']}")
        
        # 1. HÖCHSTE PRIORITÄT: Metadaten-basierte Fachsuche
        for subject in intent['subject_keywords']:
            if subject in SUBJECT_INDEX:
                print(f"   Found {len(SUBJECT_INDEX[subject])} projects in {subject} via metadata")
                for idx in SUBJECT_INDEX[subject]:
                    if idx < len(CHUNKS):
                        chunk = CHUNKS[idx]
                        # Nur Innovationsfonds-Projektseiten
                        if 'projektvorstellungen' in chunk['metadata'].get('source', '').lower():
                            if not any(r[1] == chunk for r in results):
                                results.append((200, chunk))  # HÖCHSTE PRIORITÄT!
    
    # Spezielle Behandlung für allgemeine Innovationsfonds-Anfragen
    elif intent['is_innovationsfonds_query']:
        # Priorisiere Projektvorstellungen-URLs
        for url, indices in URL_INDEX.items():
            if 'projektvorstellungen' in url:
                for idx in indices:
                    if idx < len(CHUNKS):
                        chunk = CHUNKS[idx]
                        # Check if it matches subject keywords
                        if intent['subject_keywords']:
                            chunk_subjects = [s.lower() for s in chunk['metadata'].get('fächer', [])]
                            if any(subj in chunk_subjects for subj in intent['subject_keywords']):
                                results.append((180, chunk))
                        else:
                            results.append((150, chunk))
    
    # 2. Direkte URL-Treffer
    for topic in intent['topic_keywords']:
        for url, indices in URL_INDEX.items():
            if topic in url:
                for idx in indices[:3]:
                    if idx < len(CHUNKS):
                        chunk = CHUNKS[idx]
                        if not any(r[1] == chunk for r in results):
                            results.append((100, chunk))
    
    # 3. Keyword-Index-Suche (Content-basiert)
    for topic in intent['topic_keywords'] + intent['subject_keywords']:
        if topic in KEYWORD_INDEX:
            for idx in KEYWORD_INDEX[topic][:5]:
                if idx < len(CHUNKS):
                    chunk = CHUNKS[idx]
                    if not any(r[1] == chunk for r in results):
                        results.append((80, chunk))
    
    # 4. Textsuche mit Scoring
    for i, chunk in enumerate(CHUNKS):
        if len(results) > max_results * 3:
            break
            
        content_lower = chunk['content'].lower()
        score = 0
        
        # Bonus für Innovationsfonds-Projektseiten
        if 'projektvorstellungen' in chunk['metadata'].get('source', '').lower():
            score += 30
        
        # Phrasen-Matches
        if len(query_words) > 1:
            words_list = query_lower.split()
            for j in range(len(words_list) - 1):
                phrase = f"{words_list[j]} {words_list[j+1]}"
                if phrase in content_lower:
                    score += 25
        
        # Wort-für-Wort Scoring
        content_words = set(content_lower.split())
        matching_words = query_words & content_words
        score += len(matching_words) * 5
        
        # Intent-basiertes Scoring
        if intent['is_date_query'] and any(d in content_lower for d in ['2024', '2025', '2026', 'uhr', 'datum', 'termin']):
            score += 20
        
        if intent['wants_contact'] and any(c in content_lower for c in ['anmeldung', '@', 'email', 'telefon', 'formular']):
            score += 20
            
        if intent['wants_list'] and (content_lower.count('•') > 2 or content_lower.count('\n') > 5):
            score += 15
        
        # Titel-Bonus
        if 'title' in chunk['metadata']:
            title_lower = chunk['metadata']['title'].lower()
            if any(word in title_lower for word in query_words if len(word) > 3):
                score += 30
        
        if score > 10 and not any(r[1] == chunk for r in results):
            results.append((score, chunk))
    
    # Sortiere nach Score
    results.sort(key=lambda x: x[0], reverse=True)
    
    print(f"📊 Search results before deduplication: {len(results)}")
    if results:
        print(f"   Top 5 scores: {[r[0] for r in results[:5]]}")
    
    # VERBESSERTE Deduplizierung - URL-basiert für Innovationsfonds-Projekte
    final_results = []
    seen_urls = set()
    url_count = Counter()
    
    for score, chunk in results:
        url = chunk['metadata'].get('source', '')
        
        # Für Innovationsfonds-Projektseiten: nur EINMAL pro URL (keine Duplikate!)
        is_project_page = 'projektvorstellungen' in url.lower() and 'uebersicht' in url.lower() and any(char.isdigit() for char in url.split('/')[-1])
        
        if is_project_page:
            # Prüfe ob wir diese URL schon haben
            if url not in seen_urls:
                final_results.append(chunk)
                seen_urls.add(url)
        else:
            # Normale Seiten: max 2 pro URL
            if url_count[url] < 2:
                final_results.append(chunk)
                url_count[url] += 1
            
        if len(final_results) >= max_results:
            break
    
    print(f"✅ Final results: {len(final_results)}")
    if final_results:
        for i, chunk in enumerate(final_results[:3]):
            print(f"   {i+1}. {chunk['metadata'].get('title', 'No title')}")
    
    return final_results

def create_enhanced_prompt(question: str, chunks: List[Dict], intent: Dict) -> str:
    """Erstelle optimierten Prompt mit KLAREN, KURZEN Anweisungen"""
    
    current_date = datetime.now()
    current_date_str = current_date.strftime('%d.%m.%Y')
    
    # Sortiere Events chronologisch wenn es eine Datumsabfrage ist
    if intent['is_date_query'] or any(keyword in ['workshop', 'veranstaltung'] for keyword in intent['topic_keywords']):
        sorted_events = sort_events_chronologically(chunks, current_date)
        
        context_parts = []
        
        if sorted_events['future_events']:
            context_parts.append("=== KOMMENDE VERANSTALTUNGEN (chronologisch sortiert) ===")
            for event in sorted_events['future_events']:
                days_until = (event['date'].date() - current_date.date()).days
                context_parts.append(f"\n📅 DATUM: {event['date'].strftime('%d.%m.%Y (%A)')} (in {days_until} Tagen)")
                context_parts.append(f"Quelle: {event['chunk']['metadata'].get('source', 'Unbekannt')}")
                context_parts.append(f"Titel: {event['chunk']['metadata'].get('title', 'Unbekannt')}")
                context_parts.append(event['chunk']['content'])
                context_parts.append("---")
        
        if sorted_events['past_events']:
            context_parts.append("\n\n=== VERGANGENE VERANSTALTUNGEN (bereits vorbei) ===")
            for event in sorted_events['past_events'][:5]:
                days_ago = (current_date.date() - event['date'].date()).days
                context_parts.append(f"\n📅 DATUM: {event['date'].strftime('%d.%m.%Y (%A)')} (vor {days_ago} Tagen - BEREITS VORBEI)")
                context_parts.append(f"Quelle: {event['chunk']['metadata'].get('source', 'Unbekannt')}")
                context_parts.append(f"Titel: {event['chunk']['metadata'].get('title', 'Unbekannt')}")
                context_parts.append(event['chunk']['content'])
                context_parts.append("---")
        
        if sorted_events['no_date_events']:
            context_parts.append("\n\n=== WEITERE INFORMATIONEN (ohne spezifisches Datum) ===")
            for item in sorted_events['no_date_events']:
                context_parts.append(f"\nQuelle: {item['chunk']['metadata'].get('source', 'Unbekannt')}")
                context_parts.append(f"Titel: {item['chunk']['metadata'].get('title', 'Unbekannt')}")
                context_parts.append(item['chunk']['content'])
                context_parts.append("---")
        
        context = "\n".join(context_parts)
    else:
        # Standard-Gruppierung - JEDES Projekt einzeln mit URL!
        context_parts = []
        for chunk in chunks:
            url = chunk['metadata'].get('source', 'Unbekannt')
            title = chunk['metadata'].get('title', 'Keine Beschreibung')
            faecher = chunk['metadata'].get('fächer', [])
            
            context_parts.append(f"=== Projekt: {title} ===")
            context_parts.append(f"URL: {url}")
            if faecher:
                context_parts.append(f"Fächer: {', '.join(faecher)}")
            # Nur erste 400 Zeichen - spart Token und erhält Fokus
            content_preview = chunk['content'][:400]
            context_parts.append(content_preview)
            context_parts.append("---\n")
        
        context = "\n".join(context_parts)
    
    # KOMPAKTER User-Prompt - Formatierung ist im System Prompt!
    if intent['is_innovationsfonds_query']:
        prompt = f"""KONTEXT (Innovationsfonds-Projekte mit URLs):
{context}

FRAGE: {question}

Zeige ALLE Projekte mit klickbaren Links (Format siehe System-Prompt)."""
    else:
        # Kürzerer Prompt für nicht-Projekt-Anfragen
        prompt = f"""KONTEXT:
{context}

FRAGE: {question}"""
    
    return prompt

@app.get("/")
async def root():
    return {
        "message": "DLH Chatbot API (System Prompt Version)",
        "status": "running" if len(CHUNKS) > 0 else "ERROR: No data loaded!",
        "chunks_loaded": len(CHUNKS),
        "indexed_keywords": len(KEYWORD_INDEX),
        "indexed_subjects": len(SUBJECT_INDEX),
        "version": "3.6.0"
    }

@app.get("/health")
async def health_check():
    data_status = "healthy" if len(CHUNKS) > 0 else "ERROR: No chunks loaded!"
    
    return {
        "status": data_status,
        "chunks_loaded": len(CHUNKS),
        "api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "indexed_keywords": len(KEYWORD_INDEX),
        "indexed_subjects": len(SUBJECT_INDEX),
        "subjects_available": list(SUBJECT_INDEX.keys()) if SUBJECT_INDEX else [],
        "features": "System Prompt + URL deduplication + Sonnet 4.5"
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Beantworte Fragen mit optimaler Kontext-Verarbeitung"""
    
    # Check if data is loaded
    if len(CHUNKS) == 0:
        raise HTTPException(
            status_code=500, 
            detail="Server error: No data loaded. Please check if processed/processed_chunks.json is accessible."
        )
    
    try:
        # Analysiere Intent
        intent = extract_query_intent(request.question)
        
        print(f"\n🔍 Query: {request.question}")
        print(f"   Intent: innovationsfonds={intent['is_innovationsfonds_query']}, subjects={intent['subject_keywords']}")
        
        # Führe erweiterte Suche durch
        relevant_chunks = advanced_search(
            request.question, 
            max_results=request.max_sources + 5  # Mehr Ergebnisse für Innovationsfonds
        )
        
        if not relevant_chunks:
            # Keine relevanten Chunks gefunden
            answer = f"<strong>Entschuldigung, ich konnte keine relevanten Informationen zu '{request.question}' finden.</strong><br><br>"
            answer += "Bitte versuchen Sie eine andere Formulierung oder besuchen Sie <a href='https://dlh.zh.ch' target='_blank'>dlh.zh.ch</a> für weitere Informationen."
            
            return AnswerResponse(
                question=request.question,
                answer=answer,
                sources=[]
            )
        
        # Erstelle optimierten Prompt
        prompt = create_enhanced_prompt(request.question, relevant_chunks, intent)
        
        # Get response from Claude
        try:
            # SYSTEM PROMPT für unvergessliche Formatierungsregeln
            system_prompt = """Du bist der offizielle DLH Chatbot. Antworte auf Deutsch mit HTML-Formatierung.

KRITISCHE REGEL - PROJEKTTITEL MÜSSEN IMMER KLICKBARE LINKS SEIN:
Für Innovationsfonds-Projekte gilt: JEDER Projekttitel MUSS als klickbarer Link formatiert werden.

Format: <strong><a href="VOLLSTÄNDIGE-URL" target="_blank">Projekttitel</a></strong><br>
Beschreibung in 1-2 Sätzen<br><br>

HTML-Tags:
- <br> = Zeilenumbruch
- <br><br> = Absatz zwischen Projekten  
- <strong> = Überschriften
- <a href="URL" target="_blank"> = Links
- NIEMALS Markdown (*, #, _)

BEISPIEL:
<strong><a href="https://dlh.zh.ch/home/innovationsfonds/projektvorstellungen/uebersicht/1042-histoswiss" target="_blank">HistoSwiss</a></strong><br>
Interdisziplinäres Geschichtsprojekt mit digitalen Werkzeugen<br><br>"""

            response = anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",  # Sonnet 4.5 - bestes Modell!
                max_tokens=2500,
                temperature=0.3,
                system=system_prompt,  # SYSTEM PROMPT!
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            answer = response.content[0].text
            
        except Exception as claude_error:
            print(f"🔴 Claude API Error: {claude_error}")
            print(f"🔑 API Key present: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
            
            # Besserer Fallback mit HTML-Formatierung und Links
            answer = "<strong>Entschuldigung, ich kann gerade nicht auf die KI zugreifen.</strong><br><br>"
            answer += f"Hier sind relevante Informationen zu Ihrer Frage '{request.question}':<br><br>"
            
            for i, chunk in enumerate(relevant_chunks[:5]):
                title = chunk['metadata'].get('title', 'Information')
                url = chunk['metadata'].get('source', '')
                content = chunk['content'][:300]
                content = content.replace('\n', '<br>')
                answer += f"<strong><a href='{url}' target='_blank'>{title}</a></strong><br>{content}...<br><br>"
        
        # Format sources
        sources = []
        seen_urls = set()
        for chunk in relevant_chunks[:request.max_sources]:
            url = chunk['metadata']['source']
            if url not in seen_urls:
                sources.append(Source(
                    url=url,
                    title=chunk['metadata'].get('title', 'DLH Seite'),
                    snippet=chunk['content'][:150] + "..."
                ))
                seen_urls.add(url)
        
        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Besserer Fehler-Fallback
        if 'relevant_chunks' in locals() and relevant_chunks:
            fallback_answer = f"<strong>Ein Fehler ist aufgetreten.</strong><br><br>Basierend auf den Informationen von dlh.zh.ch:<br><br>"
            
            for chunk in relevant_chunks[:3]:
                title = chunk['metadata'].get('title', 'Information')
                url = chunk['metadata'].get('source', '')
                content = chunk['content'][:200]
                
                fallback_answer += f"<strong><a href='{url}' target='_blank'>{title}</a></strong><br>{content}...<br><br>"
            
            sources = []
            for chunk in relevant_chunks[:3]:
                url = chunk['metadata']['source']
                sources.append(Source(
                    url=url,
                    title=chunk['metadata'].get('title', 'DLH Seite'),
                    snippet=chunk['content'][:150] + "..."
                ))
            
            return AnswerResponse(
                question=request.question,
                answer=fallback_answer,
                sources=sources
            )
        else:
            raise HTTPException(status_code=500, detail=str(e))

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")

if __name__ == "__main__":
    print("\n🚀 Starting DLH Chatbot API server (SYSTEM PROMPT VERSION)...")
    print("📖 API documentation: http://localhost:8000/docs")
    print("🌐 Chat interface: http://localhost:8000/static/index.html")
    print(f"📚 Loaded {len(CHUNKS)} chunks")
    print(f"🔍 Indexed {len(KEYWORD_INDEX)} keywords")
    print(f"📚 Indexed {len(SUBJECT_INDEX)} subjects in metadata")
    if SUBJECT_INDEX:
        print(f"   Subjects: {', '.join(list(SUBJECT_INDEX.keys())[:10])}...")
    print("✨ Features:")
    print("   - System Prompt für garantierte Link-Formatierung")
    print("   - URL-basierte Deduplizierung (keine Duplikate!)")
    print("   - Sonnet 4.5 (bestes Modell)")
    print("   - Metadata subject search")
    print("\n✅ All features enabled!\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
