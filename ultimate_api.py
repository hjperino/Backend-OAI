"""
Ultimate API server für DLH Chatbot - Enhanced Innovationsfonds Project Detection
NEW: Shows specific project titles with direct clickable links for Innovationsfonds queries
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
    title="DLH Chatbot API (Innovationsfonds Enhanced)",
    description="AI-powered chatbot für dlh.zh.ch mit verbesserter Innovationsfonds-Projekterkennung",
    version="3.3.0"
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
        with open('processed/processed_chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Erstelle Index für schnellere Suche
        keyword_index = {}
        url_index = {}
        
        for i, chunk in enumerate(chunks):
            # URL-basierter Index
            url = chunk['metadata'].get('source', '').lower()
            if url not in url_index:
                url_index[url] = []
            url_index[url].append(i)
            
            # ENHANCED: Erweiterte Keyword-Liste mit allen Fächern
            content = chunk['content'].lower()
            important_terms = [
                'fobizz', 'genki', 'innovationsfonds', 'cop', 'cops',
                'vernetzung', 'workshop', 'weiterbildung', 'kuratiert',
                'impuls', 'termin', 'anmeldung', 'lunch', 'learn',
                'impuls-workshop', 'impulsworkshop', 'veranstaltung', 'event',
                # Fächer
                'chemie', 'physik', 'biologie', 'mathematik', 'informatik',
                'deutsch', 'englisch', 'französisch', 'italienisch', 'spanisch',
                'geschichte', 'geografie', 'wirtschaft', 'recht', 'philosophie',
                'psychologie', 'pädagogik', 'kunst', 'musik', 'sport',
                'ethik', 'religion', 'politik'
            ]
            
            for term in important_terms:
                if term in content:
                    if term not in keyword_index:
                        keyword_index[term] = []
                    keyword_index[term].append(i)
        
        return chunks, keyword_index, url_index
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], {}, {}

# Global data storage
CHUNKS, KEYWORD_INDEX, URL_INDEX = load_and_preprocess_data()

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
    Sortiere Events chronologisch
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
    """ENHANCED: Analysiere die Absicht mit besserer Innovationsfonds-Erkennung und mehr Begriffsvarianten"""
    query_lower = query.lower()
    
    # ERWEITERT: Mehr Varianten für Innovationsfonds/Projekt-Erkennung
    innovationsfonds_terms = [
        'innovationsfonds',
        'innovationsprojekt',
        'innovationsprojekte',
        'innovations projekt',
        'innovations projekte',
        'innovation projekt',
        'innovation projekte',
        'innovatives projekt',
        'innovative projekte',
        'projekte für',
        'projekt für',
        'projekte im',
        'projekt im',
        'projekte zum',
        'projekt zum',
        'projekte',
        'welche projekte',
        'welche projekt'
    ]
    
    intent = {
        'is_date_query': any(term in query_lower for term in ['heute', 'morgen', 'termin', 'wann', 'datum', 'zeit', 'event', 'veranstaltung', 'nächste', 'kommende']),
        'is_how_to': any(term in query_lower for term in ['wie', 'anleitung', 'tutorial', 'schritte']),
        'is_definition': any(term in query_lower for term in ['was ist', 'was sind', 'definition', 'bedeutung']),
        'wants_list': any(term in query_lower for term in ['welche', 'liste', 'alle', 'überblick', 'übersicht']),
        'wants_contact': any(term in query_lower for term in ['kontakt', 'anmeldung', 'email', 'telefon', 'anmelden']),
        # NEW: Erweiterte Innovationsfonds-Projekt-Erkennung
        'is_innovationsfonds_query': any(term in query_lower for term in innovationsfonds_terms),
        'topic_keywords': []
    }
    
    # Erweiterte Themenerkennung mit allen Fächern
    topics = {
        'fobizz': ['fobizz', 'to teach', 'to-teach'],
        'genki': ['genki', 'gen ki', 'gen-ki'],
        'innovationsfonds': ['innovationsfonds', 'innovation', 'projekt'],
        'workshop': ['workshop', 'impuls', 'veranstaltung', 'impuls-workshop', 'impulsworkshop', 'event'],
        'cop': ['cop', 'cops', 'community', 'practice'],
        'weiterbildung': ['weiterbildung', 'fortbildung', 'kurs', 'schulung'],
        'vernetzung': ['vernetzung', 'netzwerk', 'austausch'],
        'kuratiert': ['kuratiert', 'kuratiertes', 'sammlung'],
        # Fächer
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
    
    return intent

def advanced_search(query: str, max_results: int = 10) -> List[Dict]:
    """ENHANCED: Verbesserte Suche mit Priorisierung von Innovationsfonds-Projektseiten"""
    intent = extract_query_intent(query)
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    results = []
    
    # Spezialbehandlung für Impuls-Workshops
    if 'impuls' in query_lower and 'workshop' in query_lower:
        intent['topic_keywords'].append('impulsworkshop')
    
    # NEW: Spezielle Behandlung für Innovationsfonds-Projektanfragen
    if intent['is_innovationsfonds_query'] or 'projekt' in query_lower:
        # Priorisiere Projektvorstellungen-URLs
        for url, indices in URL_INDEX.items():
            if 'projektvorstellungen' in url:
                # Prüfe ob ein Fach in der URL oder im Content vorkommt
                for topic in intent['topic_keywords']:
                    if topic in url or any(topic in CHUNKS[idx]['content'].lower() for idx in indices):
                        for idx in indices:
                            if idx < len(CHUNKS):
                                results.append((150, CHUNKS[idx]))  # Höchste Priorität!
    
    # 1. Direkte URL-Treffer haben hohe Priorität
    for topic in intent['topic_keywords']:
        for url, indices in URL_INDEX.items():
            if topic in url:
                for idx in indices[:3]:
                    if idx < len(CHUNKS):
                        chunk = CHUNKS[idx]
                        if not any(r[1] == chunk for r in results):
                            results.append((100, chunk))
    
    # 2. Keyword-Index-Suche
    for topic in intent['topic_keywords']:
        if topic in KEYWORD_INDEX:
            for idx in KEYWORD_INDEX[topic][:5]:
                if idx < len(CHUNKS):
                    chunk = CHUNKS[idx]
                    if not any(r[1] == chunk for r in results):
                        results.append((80, chunk))
    
    # 3. Erweiterte Textsuche mit Scoring
    for i, chunk in enumerate(CHUNKS):
        if len(results) > max_results * 3:
            break
            
        content_lower = chunk['content'].lower()
        score = 0
        
        # Bonus für Innovationsfonds-Projektseiten
        if 'projektvorstellungen' in chunk['metadata'].get('source', '').lower():
            score += 30
        
        # Exakte Phrasen-Matches
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
    
    # Diversifiziere Ergebnisse
    final_results = []
    url_count = Counter()
    
    for score, chunk in results:
        url = chunk['metadata'].get('source', '')
        # Für Innovationsfonds-Projektseiten: erlaube mehr pro URL
        max_per_url = 5 if 'projektvorstellungen' in url.lower() else 2
        
        if url_count[url] < max_per_url:
            final_results.append(chunk)
            url_count[url] += 1
            
        if len(final_results) >= max_results:
            break
    
    return final_results

def create_enhanced_prompt(question: str, chunks: List[Dict], intent: Dict) -> str:
    """ENHANCED: Erstelle Prompt mit speziellen Anweisungen für Innovationsfonds-Projekte"""
    
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
                context_parts.append(event['chunk']['content'])
                context_parts.append("---")
        
        if sorted_events['past_events']:
            context_parts.append("\n\n=== VERGANGENE VERANSTALTUNGEN (bereits vorbei) ===")
            for event in sorted_events['past_events'][:5]:
                days_ago = (current_date.date() - event['date'].date()).days
                context_parts.append(f"\n📅 DATUM: {event['date'].strftime('%d.%m.%Y (%A)')} (vor {days_ago} Tagen - BEREITS VORBEI)")
                context_parts.append(f"Quelle: {event['chunk']['metadata'].get('source', 'Unbekannt')}")
                context_parts.append(event['chunk']['content'])
                context_parts.append("---")
        
        if sorted_events['no_date_events']:
            context_parts.append("\n\n=== WEITERE INFORMATIONEN (ohne spezifisches Datum) ===")
            for item in sorted_events['no_date_events']:
                context_parts.append(f"\nQuelle: {item['chunk']['metadata'].get('source', 'Unbekannt')}")
                context_parts.append(item['chunk']['content'])
                context_parts.append("---")
        
        context = "\n".join(context_parts)
    else:
        # Standard-Gruppierung nach URL
        chunks_by_url = {}
        for chunk in chunks:
            url = chunk['metadata'].get('source', 'Unbekannt')
            title = chunk['metadata'].get('title', 'Keine Beschreibung')
            
            if url not in chunks_by_url:
                chunks_by_url[url] = {
                    'title': title,
                    'url': url,
                    'contents': []
                }
            chunks_by_url[url]['contents'].append(chunk['content'])
        
        context_parts = []
        for url, data in chunks_by_url.items():
            context_parts.append(f"=== Projekt: {data['title']} ===")
            context_parts.append(f"URL: {url}")
            for content in data['contents']:
                context_parts.append(content)
            context_parts.append("")
        
        context = "\n\n".join(context_parts)
    
    # Intent-spezifische Anweisungen
    intent_instructions = ""
    
    # NEW: Spezielle Anweisungen für Innovationsfonds-Projektanfragen
    if intent['is_innovationsfonds_query'] or 'projekt' in question.lower():
        intent_instructions += """
🎯 INNOVATIONSFONDS-PROJEKTE - WICHTIGE FORMATIERUNGSREGELN:

1. PROJEKTTITEL UND LINKS:
   - Zeige JEDEN Projekttitel als <strong>Überschrift</strong>
   - Mache JEDEN Projekttitel zu einem klickbaren Link zur Projektseite
   - Format: <strong><a href="VOLLSTÄNDIGE-URL" target="_blank">Projekttitel</a></strong>
   
2. PROJEKTBESCHREIBUNG:
   - Gib eine kurze Beschreibung unter jedem Projekttitel
   - Verwende <br> für Zeilenumbrüche
   
3. BEISPIEL FÜR PERFEKTE FORMATIERUNG:
   <strong><a href="https://dlh.zh.ch/home/innovationsfonds/projektvorstellungen/uebersicht/735-kriminalistik-als-werkstattunterricht-zum-thema-trennmethoden" target="_blank">Kriminalistik als Werkstattunterricht, zum Thema Trennmethoden</a></strong><br>
   Mit chemischen Trennmethoden den Verbrecher:innen auf die Spur kommen<br><br>
   
   <strong><a href="https://dlh.zh.ch/home/innovationsfonds/projektvorstellungen/uebersicht/556-molekularvisualisierung-chemie" target="_blank">Molekularvisualisierung Chemie</a></strong><br>
   Bindigkeit, Geometrie und zwischenmolekulare Kräfte von Molekülen im computergestützten Unterricht<br><br>

4. WICHTIG:
   - Verwende die VOLLSTÄNDIGE URL aus dem Kontext (mit https://dlh.zh.ch...)
   - Liste ALLE gefundenen Projekte auf
   - Sortiere nicht alphabetisch, sondern nach Relevanz aus dem Kontext
   - Füge am Ende KEINE generischen Listen hinzu (wie "Chemielabor, Molekularvisualisierung...")
"""
    
    if intent['is_date_query']:
        intent_instructions += f"""
TERMINE UND VERANSTALTUNGEN:
- Heutiges Datum: {current_date_str}
- Die Events sind chronologisch sortiert
- Formatierung: <br>• <strong>DD.MM.YYYY (Wochentag)</strong> - Uhrzeit - Titel
- Markiere vergangene Events: <em>(bereits vorbei)</em>
- Zeige Anmeldelinks: <a href="URL" target="_blank">Hier anmelden</a>
"""
    
    if intent['wants_list']:
        intent_instructions += """
LISTEN UND ÜBERSICHTEN:
- Vollständige, strukturierte Listen
- <strong>Überschriften</strong> für Kategorien
- <br>• für Hauptpunkte
- <br>&nbsp;&nbsp;→ für Unterpunkte
- ALLE gefundenen Elemente zeigen
"""
    
    if intent['wants_contact']:
        intent_instructions += """
KONTAKT UND ANMELDUNG:
- Alle Kontaktinformationen angeben
- Links: <a href="URL" target="_blank">Linktext</a>
- E-Mails: <a href="mailto:email@domain.ch">email@domain.ch</a>
- Telefon: <strong>Tel: +41 XX XXX XX XX</strong>
"""
    
    prompt = f"""Du bist der offizielle KI-Assistent des Digital Learning Hub (DLH) Zürich.
Beantworte die folgende Frage präzise und vollständig basierend auf den bereitgestellten Informationen.

WICHTIGE REGELN:
1. Verwende NUR Informationen aus dem bereitgestellten Kontext
2. Sei spezifisch und vollständig - liste ALLE relevanten Informationen auf
3. Wenn etwas nicht im Kontext steht, sage das klar
4. Verweise bei Bedarf auf die DLH-Website für weitere Informationen
5. Bei Anmeldelinks: IMMER als klickbare Links formatieren

FORMATIERUNG (SEHR WICHTIG für HTML-Darstellung):
- Verwende KEINE Markdown-Zeichen (*, #, _, -)
- Verwende <br><br> für Absätze zwischen Abschnitten
- Verwende <br> für Zeilenumbrüche innerhalb von Listen
- Verwende <strong>Text</strong> für Überschriften und wichtige Begriffe
- Verwende <em>Text</em> für Hervorhebungen
- Strukturiere Listen mit <br>• für Hauptpunkte
- Verwende <br>&nbsp;&nbsp;→ für Unterpunkte
- Mache URLs klickbar: <a href="URL" target="_blank">Linktext</a>
- E-Mails: <a href="mailto:email@domain.ch">email@domain.ch</a>

{intent_instructions}

KONTEXT AUS DER DLH-WEBSITE:
{context}

FRAGE: {question}

Erstelle eine hilfreiche, gut strukturierte und vollständige Antwort mit perfekter HTML-Formatierung:"""
    
    return prompt

@app.get("/")
async def root():
    return {
        "message": "DLH Chatbot API (Innovationsfonds Enhanced)",
        "status": "running",
        "chunks_loaded": len(CHUNKS),
        "indexed_keywords": len(KEYWORD_INDEX),
        "version": "3.3.0 - Enhanced Innovationsfonds project detection"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chunks_loaded": len(CHUNKS),
        "api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "indexed_keywords": len(KEYWORD_INDEX),
        "features": "Enhanced Innovationsfonds project links"
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Beantworte Fragen mit optimaler Innovationsfonds-Projekterkennung und Debug-Ausgabe"""
    try:
        # Analysiere Intent
        intent = extract_query_intent(request.question)
        
        # DEBUG OUTPUT
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG - Query Analysis")
        print(f"{'='*60}")
        print(f"Question: {request.question}")
        print(f"is_innovationsfonds_query: {intent['is_innovationsfonds_query']}")
        print(f"topic_keywords: {intent['topic_keywords']}")
        print(f"{'='*60}\n")
        
        # Führe erweiterte Suche durch
        relevant_chunks = advanced_search(
            request.question, 
            max_results=request.max_sources + 5  # Mehr für Innovationsfonds-Projekte
        )
        
        # DEBUG OUTPUT
        print(f"🔍 DEBUG - Search Results:")
        print(f"   Total chunks found: {len(relevant_chunks)}")
        projekt_urls = [c['metadata'].get('source', '') 
                        for c in relevant_chunks 
                        if 'projektvorstellungen' in c['metadata'].get('source', '').lower()]
        print(f"   Projektvorstellungen URLs: {len(projekt_urls)}")
        if projekt_urls:
            print(f"   Sample URLs:")
            for url in projekt_urls[:3]:
                print(f"      - {url}")
        else:
            print(f"   ⚠️  NO projektvorstellungen URLs found!")
            print(f"   Sample of found URLs:")
            for chunk in relevant_chunks[:3]:
                print(f"      - {chunk['metadata'].get('source', 'No URL')}")
        print(f"{'='*60}\n")
        
        # Erstelle optimierten Prompt
        prompt = create_enhanced_prompt(request.question, relevant_chunks, intent)
        
        # Get response from Claude
        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,  # Erhöht für längere Projektlisten
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            answer = response.content[0].text
            
        except Exception as claude_error:
            print(f"🔴 Claude API Error: {claude_error}")
            print(f"🔑 API Key present: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
            
            # Fallback mit HTML-Formatierung
            answer = "<strong>Entschuldigung, ich kann gerade nicht auf die KI zugreifen.</strong><br><br>"
            answer += f"Hier sind relevante Informationen zu Ihrer Frage '{request.question}':<br><br>"
            
            for i, chunk in enumerate(relevant_chunks[:3]):
                title = chunk['metadata'].get('title', 'Information')
                url = chunk['metadata'].get('source', '#')
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
        # Fehler-Fallback
        if relevant_chunks:
            fallback_answer = f"<strong>Ein Fehler ist aufgetreten.</strong><br><br>"
            url = relevant_chunks[0]['metadata'].get('source', '#')
            title = relevant_chunks[0]['metadata'].get('title', 'Information')
            fallback_answer += f"<strong><a href='{url}' target='_blank'>{title}</a></strong><br>"
            fallback_answer += f"{relevant_chunks[0]['content'][:300]}..."
            
            sources = [Source(
                url=url,
                title=title,
                snippet=relevant_chunks[0]['content'][:150] + "..."
            )]
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
except:
    pass  # Frontend directory might not exist

if __name__ == "__main__":
    print("\n🚀 Starting DLH Chatbot API (Innovationsfonds Enhanced)...")
    print("📖 API documentation: http://localhost:8000/docs")
    print("🌐 Frontend: https://perino.info/dlh-chatbot")
    print(f"📚 Loaded {len(CHUNKS)} chunks")
    print(f"🔍 Indexed {len(KEYWORD_INDEX)} keywords")
    print("✨ NEW: Enhanced Innovationsfonds project detection with direct links!")
    print("\n✅ All features enabled!\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
