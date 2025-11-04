"""
Ultimate API server für DLH Chatbot - Verbesserte Version mit chronologischer Event-Sortierung
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
    title="DLH Chatbot API (Ultimate Improved)",
    description="AI-powered chatbot für dlh.zh.ch mit verbesserter Event-Sortierung",
    version="3.1.0"
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
            
            # Keyword-Index für wichtige Begriffe
            content = chunk['content'].lower()
            important_terms = [
                'fobizz', 'genki', 'innovationsfonds', 'cop', 'cops',
                'vernetzung', 'workshop', 'weiterbildung', 'kuratiert',
                'impuls', 'termin', 'anmeldung', 'lunch', 'learn',
                'impuls-workshop', 'impulsworkshop', 'veranstaltung', 'event',
                'one change', 'mintwoch', 'call', 'reihe', 'inputorientiert'  # inputorientiert = Impuls-Workshop Kategorie!
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
    Extrahiere Daten aus Text und gib sie mit dem ursprünglichen Text zurück
    Unterstützt Formate: DD.MM.YYYY, DD.MM.YY, DD. Monat YYYY
    """
    dates_found = []
    
    # Regex-Muster für verschiedene Datumsformate
    patterns = [
        # DD.MM.YYYY oder DD.MM.YY
        r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})',
        # DD. Monat YYYY (mit oder ohne Punkt nach Monat)
        r'(\d{1,2})\.\s*(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\.?\s*(\d{4})',
    ]
    
    month_map = {
        'januar': 1, 'februar': 2, 'märz': 3, 'april': 4,
        'mai': 5, 'juni': 6, 'juli': 7, 'august': 8,
        'september': 9, 'oktober': 10, 'november': 11, 'dezember': 12
    }
    
    # Suche nach DD.MM.YYYY Format
    for match in re.finditer(patterns[0], text):
        try:
            day = int(match.group(1))
            month = int(match.group(2))
            year_str = match.group(3)
            year = int(year_str) if len(year_str) == 4 else (2000 + int(year_str))
            
            date_obj = datetime(year, month, day)
            # Extrahiere Kontext um das Datum (ca. 100 Zeichen vor und nach)
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].strip()
            
            dates_found.append((date_obj, context, match.group(0)))
        except ValueError:
            continue
    
    # Suche nach DD. Monat YYYY Format
    for match in re.finditer(patterns[1], text, re.IGNORECASE):
        try:
            day = int(match.group(1))
            month_name = match.group(2).lower()
            month = month_map.get(month_name)
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
    
    Returns:
        Dict mit 'future_events', 'past_events' und 'no_date_events'
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
            # Sortiere Daten innerhalb des Chunks
            dates.sort(key=lambda x: x[0])
            
            # Nimm das früheste Datum als Referenz für diesen Chunk
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
    
    # Sortiere zukünftige Events: nächstes Datum zuerst
    future_events.sort(key=lambda x: x['date'])
    
    # Sortiere vergangene Events: neuestes zuerst
    past_events.sort(key=lambda x: x['date'], reverse=True)
    
    return {
        'future_events': future_events,
        'past_events': past_events,
        'no_date_events': no_date_events
    }

def extract_query_intent(query: str) -> Dict[str, any]:
    """Analysiere die Absicht der Frage"""
    query_lower = query.lower()
    
    intent = {
        'is_date_query': any(term in query_lower for term in ['heute', 'morgen', 'termin', 'wann', 'datum', 'zeit', 'event', 'veranstaltung']),
        'is_how_to': any(term in query_lower for term in ['wie', 'anleitung', 'tutorial', 'schritte']),
        'is_definition': any(term in query_lower for term in ['was ist', 'was sind', 'definition', 'bedeutung']),
        'wants_list': any(term in query_lower for term in ['welche', 'liste', 'alle', 'überblick', 'übersicht']),
        'wants_contact': any(term in query_lower for term in ['kontakt', 'anmeldung', 'email', 'telefon', 'anmelden']),
        'topic_keywords': [],
        'is_impulsworkshop_query': 'impulsworkshop' in query_lower or 'impuls-workshop' in query_lower  # Spezielle Flag für Impulsworkshop-Anfragen
    }
    
    # Erweiterte Themenerkennung
    topics = {
        'fobizz': ['fobizz', 'to teach', 'to-teach'],
        'genki': ['genki', 'gen ki', 'gen-ki'],
        'innovationsfonds': ['innovationsfonds', 'innovation', 'projekt'],
        'workshop': ['workshop', 'impuls', 'veranstaltung', 'impuls-workshop', 'impulsworkshop', 'event', 'one change', 'mintwoch', 'call', 'reihe', 'inputorientiert'],
        'cop': ['cop', 'cops', 'community', 'practice'],
        'weiterbildung': ['weiterbildung', 'fortbildung', 'kurs', 'schulung'],
        'vernetzung': ['vernetzung', 'netzwerk', 'austausch'],
        'kuratiert': ['kuratiert', 'kuratiertes', 'sammlung']
    }
    
    for topic, keywords in topics.items():
        if any(kw in query_lower for kw in keywords):
            intent['topic_keywords'].append(topic)
    
    return intent

def advanced_search(query: str, max_results: int = 8) -> List[Dict]:
    """Verbesserte Suche mit Intent-Analyse und Ranking"""
    intent = extract_query_intent(query)
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    results = []
    
    # Spezialbehandlung für Impuls-Workshops
    if 'impuls' in query_lower and 'workshop' in query_lower:
        intent['topic_keywords'].append('impulsworkshop')
    
    # 1. Direkte URL-Treffer haben höchste Priorität
    for topic in intent['topic_keywords']:
        for url, indices in URL_INDEX.items():
            if topic in url:
                for idx in indices[:3]:  # Top 3 von jeder passenden URL
                    if idx < len(CHUNKS):
                        results.append((100, CHUNKS[idx]))
    
    # 2. Keyword-Index-Suche
    for topic in intent['topic_keywords']:
        if topic in KEYWORD_INDEX:
            for idx in KEYWORD_INDEX[topic][:5]:  # Top 5 pro Keyword
                if idx < len(CHUNKS):
                    chunk = CHUNKS[idx]
                    if not any(r[1] == chunk for r in results):  # Keine Duplikate
                        results.append((80, chunk))
    
    # 3. Erweiterte Textsuche mit Scoring
    for i, chunk in enumerate(CHUNKS):
        if len(results) > max_results * 2:  # Früh abbrechen wenn genug
            break
            
        content_lower = chunk['content'].lower()
        score = 0
        
        # Exakte Phrasen-Matches (sehr wichtig!)
        if len(query_words) > 1:
            # 2-Wort-Phrasen
            words_list = query_lower.split()
            for j in range(len(words_list) - 1):
                phrase = f"{words_list[j]} {words_list[j+1]}"
                if phrase in content_lower:
                    score += 25
        
        # Wort-für-Wort Scoring
        content_words = set(content_lower.split())
        matching_words = query_words & content_words
        score += len(matching_words) * 5
        
        # Spezial-Bonus für Impulsworkshop-Anfragen
        if intent.get('is_impulsworkshop_query') and 'inputorientiert' in content_lower:
            score += 50  # Sehr hoher Bonus für Chunks mit der Impuls-Workshop Kategorie
        
        # Intent-basiertes Scoring
        if intent['is_date_query'] and any(d in content_lower for d in ['2024', '2025', '2026', 'uhr', 'datum', 'termin']):
            score += 30  # Erhöht von 20
        
        # Extra Bonus für November/Dezember 2025 Events (aktueller Monat)
        if 'november 2025' in content_lower or 'dezember 2025' in content_lower:
            score += 25
        
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
    
    # Diversifiziere Ergebnisse - nicht zu viele von der gleichen URL
    final_results = []
    url_count = Counter()
    
    for score, chunk in results:
        url = chunk['metadata'].get('source', '')
        if url_count[url] < 3:  # Max 3 Chunks pro URL
            final_results.append(chunk)
            url_count[url] += 1
            
        if len(final_results) >= max_results:
            break
    
    return final_results

def create_enhanced_prompt(question: str, chunks: List[Dict], intent: Dict) -> str:
    """Erstelle einen optimierten Prompt mit chronologischer Event-Sortierung"""
    
    current_date = datetime.now()
    current_date_str = current_date.strftime('%d.%m.%Y')
    
    # Sortiere Events chronologisch wenn es eine Datumsabfrage ist
    if intent['is_date_query'] or any(keyword in ['workshop', 'veranstaltung'] for keyword in intent['topic_keywords']):
        sorted_events = sort_events_chronologically(chunks, current_date)
        
        # Erstelle strukturierten Event-Kontext
        context_parts = []
        
        # Zukünftige Events
        if sorted_events['future_events']:
            context_parts.append("=== KOMMENDE VERANSTALTUNGEN (chronologisch sortiert) ===")
            for event in sorted_events['future_events']:
                days_until = (event['date'].date() - current_date.date()).days
                context_parts.append(f"\n📅 DATUM: {event['date'].strftime('%d.%m.%Y (%A)')} (in {days_until} Tagen)")
                context_parts.append(f"Quelle: {event['chunk']['metadata'].get('source', 'Unbekannt')}")
                context_parts.append(event['chunk']['content'])
                context_parts.append("---")
        
        # Vergangene Events
        if sorted_events['past_events']:
            context_parts.append("\n\n=== VERGANGENE VERANSTALTUNGEN (bereits vorbei) ===")
            for event in sorted_events['past_events'][:5]:  # Maximal 5 vergangene Events
                days_ago = (current_date.date() - event['date'].date()).days
                context_parts.append(f"\n📅 DATUM: {event['date'].strftime('%d.%m.%Y (%A)')} (vor {days_ago} Tagen - BEREITS VORBEI)")
                context_parts.append(f"Quelle: {event['chunk']['metadata'].get('source', 'Unbekannt')}")
                context_parts.append(event['chunk']['content'])
                context_parts.append("---")
        
        # Events ohne erkennbares Datum
        if sorted_events['no_date_events']:
            context_parts.append("\n\n=== WEITERE INFORMATIONEN (ohne spezifisches Datum) ===")
            for item in sorted_events['no_date_events']:
                context_parts.append(f"\nQuelle: {item['chunk']['metadata'].get('source', 'Unbekannt')}")
                context_parts.append(item['chunk']['content'])
                context_parts.append("---")
        
        context = "\n".join(context_parts)
    else:
        # Standard-Gruppierung nach URL für nicht-Event-Anfragen
        chunks_by_url = {}
        for chunk in chunks:
            url = chunk['metadata'].get('source', 'Unbekannt')
            if url not in chunks_by_url:
                chunks_by_url[url] = []
            chunks_by_url[url].append(chunk['content'])
        
        context_parts = []
        for url, contents in chunks_by_url.items():
            context_parts.append(f"=== Quelle: {url} ===")
            for content in contents:
                context_parts.append(content)
            context_parts.append("")
        
        context = "\n\n".join(context_parts)
    
    # Intent-spezifische Anweisungen
    intent_instructions = ""
    
    if intent['is_date_query']:
        intent_instructions += f"""
TERMINE UND VERANSTALTUNGEN - WICHTIGE SORTIERUNGSREGELN:
- Heutiges Datum: {current_date_str}
- Die Events sind bereits chronologisch sortiert (nächstes Datum zuerst)
- BEHALTE diese chronologische Reihenfolge bei!
- Präsentiere ZUERST kommende Events, dann vergangene Events
- Formatierung für Events:
  <br>• <strong>DD.MM.YYYY (Wochentag)</strong> - Uhrzeit - Veranstaltungstitel
  <br>&nbsp;&nbsp;→ Kurzbeschreibung falls vorhanden
  <br>&nbsp;&nbsp;→ Anmeldung: <a href="URL" target="_blank">Hier anmelden</a>
  
- Markiere vergangene Events deutlich:
  <br><br><strong>⚠️ Bereits vergangene Veranstaltungen:</strong><br>
  • <strong>DD.MM.YYYY</strong> - Titel <em>(bereits vorbei)</em>

- Zeige IMMER Anmeldelinks wenn vorhanden
- Bei mehreren Events an einem Tag: gruppiere sie zusammen
"""
    
    if intent['wants_list']:
        intent_instructions += """
LISTEN UND ÜBERSICHTEN:
- Erstelle eine vollständige, strukturierte Liste
- Verwende klare Kategorien mit <strong>Überschriften</strong>
- Nutze <br>• für Aufzählungspunkte
- Nutze <br>&nbsp;&nbsp;→ für Unterpunkte
- Zeige ALLE gefundenen Elemente, nicht nur Beispiele
- Strukturiere logisch (z.B. nach Kategorien oder chronologisch)
"""
    
    if intent['wants_contact']:
        intent_instructions += """
KONTAKT UND ANMELDUNG:
- Gib ALLE gefundenen Kontaktinformationen an
- Mache Links klickbar: <a href="URL" target="_blank">Linktext</a>
- Betone wichtige Informationen wie Anmeldefristen
- Zeige E-Mail-Adressen als Links: <a href="mailto:email@domain.ch">email@domain.ch</a>
- Telefonnummern: <strong>Tel: +41 XX XXX XX XX</strong>
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
- Verwende <em>Text</em> für Hervorhebungen (z.B. "bereits vorbei")
- Strukturiere Listen mit <br>• für Hauptpunkte
- Verwende <br>&nbsp;&nbsp;→ für Unterpunkte
- Mache URLs klickbar: <a href="URL" target="_blank">Linktext</a>
- E-Mails: <a href="mailto:email@domain.ch">email@domain.ch</a>

Beispiel für perfekte Event-Formatierung:
<strong>Kommende Veranstaltungen</strong><br><br>

<strong>📅 November 2025</strong><br>
• <strong>19.11.2025 (Dienstag)</strong> - 12:15 - 13:00 Uhr - Lunch & Learn: KI im Unterricht<br>
&nbsp;&nbsp;→ Online-Format via Zoom<br>
&nbsp;&nbsp;→ Anmeldung: <a href="https://example.ch/anmeldung" target="_blank">Hier zur Anmeldung</a><br>
<br>
• <strong>26.11.2025 (Dienstag)</strong> - 14:00 - 15:30 Uhr - Impuls-Workshop: Digitale Tools<br>
&nbsp;&nbsp;→ Präsenz im DLH, Raum 3.14<br>
&nbsp;&nbsp;→ Anmeldung: <a href="https://example.ch/anmeldung2" target="_blank">Hier zur Anmeldung</a><br><br>

<strong>📅 Dezember 2025</strong><br>
• <strong>03.12.2025 (Mittwoch)</strong> - 09:00 - 10:00 Uhr - Sprechstunde<br><br>

<strong>⚠️ Bereits vergangene Veranstaltungen:</strong><br>
• <strong>15.10.2025</strong> - Workshop Fobizz <em>(bereits vorbei)</em><br>
• <strong>22.09.2025</strong> - Kick-off Meeting <em>(bereits vorbei)</em>

{intent_instructions}

KONTEXT AUS DER DLH-WEBSITE:
{context}

FRAGE: {question}

Erstelle eine hilfreiche, gut strukturierte und vollständige Antwort mit perfekter HTML-Formatierung:"""
    
    return prompt

@app.get("/")
async def root():
    return {
        "message": "DLH Chatbot API (Ultimate Improved)",
        "status": "running",
        "chunks_loaded": len(CHUNKS),
        "indexed_keywords": len(KEYWORD_INDEX),
        "version": "3.1.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chunks_loaded": len(CHUNKS),
        "api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "indexed_keywords": len(KEYWORD_INDEX)
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Beantworte Fragen mit optimaler Kontext-Verarbeitung und Event-Sortierung"""
    relevant_chunks = []  # Initialize to avoid UnboundLocalError
    
    try:
        # Analysiere Intent
        intent = extract_query_intent(request.question)
        
        # Führe erweiterte Suche durch
        relevant_chunks = advanced_search(
            request.question, 
            max_results=request.max_sources + 3
        )
        
        # Erstelle optimierten Prompt
        prompt = create_enhanced_prompt(request.question, relevant_chunks, intent)
        
        # Get response from Claude
        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,  # Erhöht für ausführlichere Event-Listen
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
            print(f"🔑 API Key starts with: {os.getenv('ANTHROPIC_API_KEY', 'NOT_SET')[:10]}...")
            
            # Besserer Fallback mit HTML-Formatierung
            answer = "<strong>Entschuldigung, ich kann gerade nicht auf die KI zugreifen.</strong><br><br>"
            answer += f"Hier sind relevante Informationen zu Ihrer Frage '{request.question}':<br><br>"
            
            for i, chunk in enumerate(relevant_chunks[:3]):
                title = chunk['metadata'].get('title', 'Information')
                content = chunk['content'][:400]
                content = content.replace('\n', '<br>')
                answer += f"<strong>{title}:</strong><br>{content}...<br><br>"
        
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
        # Besserer Fehler-Fallback
        if relevant_chunks:
            fallback_answer = f"<strong>Ein Fehler ist aufgetreten.</strong><br><br>Basierend auf den Informationen von dlh.zh.ch:<br><br>{relevant_chunks[0]['content'][:300]}..."
            sources = [Source(
                url=relevant_chunks[0]['metadata']['source'],
                title=relevant_chunks[0]['metadata']['title'],
                snippet=relevant_chunks[0]['content'][:150] + "..."
            )]
            return AnswerResponse(
                question=request.question,
                answer=fallback_answer,
                sources=sources
            )
        else:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n🚀 Starting Ultimate DLH Chatbot API server (IMPROVED VERSION)...")
    print("📝 API documentation: http://localhost:8000/docs")
    print("🌐 Frontend hosted at: https://perino.info/dlh-chatbot")
    print(f"📚 Loaded {len(CHUNKS)} chunks")
    print(f"🔍 Indexed {len(KEYWORD_INDEX)} keywords")
    print("✨ NEW: Chronological event sorting with past/future separation!")
    print("\n✅ All features enabled!\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
