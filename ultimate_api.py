"""
Ultimate API server für DLH Chatbot - Kombiniert alle Verbesserungen
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
from datetime import datetime
from collections import Counter

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="DLH Chatbot API (Ultimate)",
    description="AI-powered chatbot für dlh.zh.ch mit allen Optimierungen",
    version="3.0.0"
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
        with open('data/processed/processed_chunks.json', 'r', encoding='utf-8') as f:
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
                'impuls-workshop', 'impulsworkshop'  # Zusätzlich für Impuls-Workshops
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

def extract_query_intent(query: str) -> Dict[str, any]:
    """Analysiere die Absicht der Frage"""
    query_lower = query.lower()
    
    intent = {
        'is_date_query': any(term in query_lower for term in ['heute', 'morgen', 'termin', 'wann', 'datum', 'zeit']),
        'is_how_to': any(term in query_lower for term in ['wie', 'anleitung', 'tutorial', 'schritte']),
        'is_definition': any(term in query_lower for term in ['was ist', 'was sind', 'definition', 'bedeutung']),
        'wants_list': any(term in query_lower for term in ['welche', 'liste', 'alle', 'überblick', 'übersicht']),
        'wants_contact': any(term in query_lower for term in ['kontakt', 'anmeldung', 'email', 'telefon', 'anmelden']),
        'topic_keywords': []
    }
    
    # Erweiterte Themenerkennung
    topics = {
        'fobizz': ['fobizz', 'to teach', 'to-teach'],
        'genki': ['genki', 'gen ki', 'gen-ki'],
        'innovationsfonds': ['innovationsfonds', 'innovation', 'projekt'],
        'workshop': ['workshop', 'impuls', 'veranstaltung', 'impuls-workshop', 'impulsworkshop'],
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
                    score += 25  # Erhöht von 20
        
        # Wort-für-Wort Scoring
        content_words = set(content_lower.split())
        matching_words = query_words & content_words
        score += len(matching_words) * 5
        
        # Intent-basiertes Scoring
        if intent['is_date_query'] and any(d in content_lower for d in ['2024', '2025', 'uhr', 'datum', 'termin']):
            score += 20  # Erhöht von 15
        
        if intent['wants_contact'] and any(c in content_lower for c in ['anmeldung', '@', 'email', 'telefon', 'formular']):
            score += 20
            
        if intent['wants_list'] and (content_lower.count('•') > 2 or content_lower.count('\n') > 5):
            score += 15
        
        # Titel-Bonus
        if 'title' in chunk['metadata']:
            title_lower = chunk['metadata']['title'].lower()
            if any(word in title_lower for word in query_words if len(word) > 3):
                score += 30  # Erhöht von 25
        
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
    """Erstelle einen optimierten Prompt basierend auf Intent und mit HTML-Formatierung"""
    
    # Gruppiere Chunks nach URL für bessere Übersicht
    chunks_by_url = {}
    for chunk in chunks:
        url = chunk['metadata'].get('source', 'Unbekannt')
        if url not in chunks_by_url:
            chunks_by_url[url] = []
        chunks_by_url[url].append(chunk['content'])
    
    # Erstelle strukturierten Kontext
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
        intent_instructions += """
TERMINE UND DATEN:
- Heutiges Datum: {date}
- Liste ALLE gefundenen Termine chronologisch auf
- Verwende das Format: <br>• <strong>Datum (Wochentag)</strong> - Zeit - Veranstaltung
- Hebe vergangene Termine als "bereits vorbei" hervor
- Zeige IMMER die Anmeldelinks wenn vorhanden
""".format(date=datetime.now().strftime('%d.%m.%Y'))
    
    if intent['wants_list']:
        intent_instructions += """
LISTEN UND ÜBERSICHTEN:
- Erstelle eine vollständige, strukturierte Liste
- Verwende klare Kategorien mit <strong>Überschriften</strong>
- Nutze <br>• für Aufzählungspunkte
- Zeige ALLE gefundenen Elemente, nicht nur Beispiele
"""
    
    if intent['wants_contact']:
        intent_instructions += """
KONTAKT UND ANMELDUNG:
- Gib ALLE gefundenen Kontaktinformationen an
- Mache Links klickbar: <a href="URL" target="_blank">Linktext</a>
- Betone wichtige Informationen wie Anmeldefristen
- Zeige E-Mail-Adressen und Telefonnummern deutlich
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
- Verwende <strong>Text</strong> für Überschriften
- Strukturiere Listen mit <br>• für jeden Punkt
- Mache URLs klickbar: <a href="URL" target="_blank">Linktext</a>
- E-Mails: <a href="mailto:email@domain.ch">email@domain.ch</a>

Beispiel für gute Formatierung:
<strong>Überschrift</strong><br><br>
Hier ist der einführende Text für diesen Abschnitt.<br><br>
<strong>Wichtige Punkte</strong><br>
• Erster wichtiger Punkt<br>
• Zweiter wichtiger Punkt<br>
• Dritter wichtiger Punkt<br><br>
<strong>Termine</strong><br>
• <strong>19.11.2025:</strong> 12:15 - 13:00 Uhr - Lunch & Learn<br>
• <strong>26.11.2025:</strong> 09:00 - 10:00 Uhr - Sprechstunde<br><br>
<strong>Anmeldung</strong><br>
<a href="https://anmeldelink.ch" target="_blank">Hier zur Anmeldung</a>

{intent_instructions}

KONTEXT AUS DER DLH-WEBSITE:
{context}

FRAGE: {question}

Erstelle eine hilfreiche, gut strukturierte und vollständige Antwort:"""
    
    return prompt

@app.get("/")
async def root():
    return {
        "message": "DLH Chatbot API (Ultimate)",
        "status": "running",
        "chunks_loaded": len(CHUNKS),
        "indexed_keywords": len(KEYWORD_INDEX)
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
    """Beantworte Fragen mit optimaler Kontext-Verarbeitung"""
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
                model="claude-sonnet-4-20250514",  # Neuestes Modell
                max_tokens=1500,  # Mehr Tokens für ausführlichere Antworten
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
                # Füge Zeilenumbrüche für bessere Lesbarkeit hinzu
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

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

if __name__ == "__main__":
    print("\n🚀 Starting Ultimate DLH Chatbot API server...")
    print("📝 API documentation: http://localhost:8000/docs")
    print("🌐 Chat interface: http://localhost:8000/static/index.html")
    print(f"📚 Loaded {len(CHUNKS)} chunks")
    print(f"🔍 Indexed {len(KEYWORD_INDEX)} keywords")
    print("\n✅ All features enabled!\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
