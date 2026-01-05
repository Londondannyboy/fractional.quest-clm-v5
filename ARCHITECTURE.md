# VIC CLM Architecture Guide

A comprehensive guide to building a Custom Language Model (CLM) for Hume EVI voice assistants.

---

## Overview

VIC is a voice-enabled AI historian that answers questions about London's hidden history. It uses:
- **Hume EVI** for voice interaction (speech-to-text, text-to-speech)
- **Custom Language Model (CLM)** for response generation
- **Groq** for fast LLM inference
- **Neon** for article storage (PostgreSQL + pgvector)
- **Zep** for conversation memory and knowledge graphs
- **Voyage AI** for embeddings

```
┌─────────────────────────────────────────────────────────────────┐
│                         User (Voice)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Hume EVI                                │
│  - Speech-to-text (transcription)                               │
│  - Emotion detection                                            │
│  - Text-to-speech (voice synthesis)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ POST /chat/completions (OpenAI format)
┌─────────────────────────────────────────────────────────────────┐
│                    VIC CLM (This Project)                       │
│  - FastAPI server on Vercel                                     │
│  - Streaming SSE responses                                      │
│  - Dual-path architecture (fast + enrichment)                   │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│    Groq     │      │    Neon     │      │     Zep     │
│  LLM API    │      │  Articles   │      │   Memory    │
│  (Llama 3)  │      │  + vectors  │      │   + Graph   │
└─────────────┘      └─────────────┘      └─────────────┘
```

---

## File Structure

```
vic-clm/
├── api/
│   ├── index.py          # FastAPI app, /chat/completions endpoint
│   ├── agent.py          # Response generation, session context
│   ├── agent_config.py   # System prompts, agent configuration
│   ├── agent_deps.py     # Pydantic AI dependencies dataclass
│   ├── models.py         # Pydantic models for requests/responses
│   ├── tools.py          # Search, embeddings, entity extraction
│   └── database.py       # Neon PostgreSQL connection
├── vercel.json           # Vercel deployment config
├── requirements.txt      # Python dependencies
└── ARCHITECTURE.md       # This file
```

---

## Key Components

### 1. FastAPI Endpoint (`api/index.py`)

The main entry point. Hume EVI calls this endpoint with conversation history.

```python
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # Extract the user's message
    user_message = extract_user_message(messages)

    # Handle special cases
    if is_greeting_request:
        return StreamingResponse(stream_response(greeting))

    if not user_message:  # Silence
        return Response(status_code=204)

    # Check for affirmations ("yes" to a suggestion)
    is_affirm, topic_hint = is_affirmation(user_message)
    if is_affirm:
        actual_query = get_last_suggestion(session_id) or user_message

    # Stream response with filler phrases
    return StreamingResponse(
        stream_with_padding(actual_query, session_id, user_name),
        media_type="text/event-stream"
    )
```

### 2. Streaming Response Format

Hume EVI expects OpenAI-compatible SSE (Server-Sent Events):

```python
async def stream_response(text: str, session_id: str):
    chunk_id = str(uuid4())
    created = int(time.time())

    # Stream token by token
    tokens = enc.encode(text)
    for i, token_id in enumerate(tokens):
        token_text = enc.decode([token_id])

        chunk = ChatCompletionChunk(
            id=chunk_id,
            choices=[Choice(
                delta=ChoiceDelta(
                    content=token_text,
                    role="assistant" if i == 0 else None
                ),
                index=0
            )],
            created=created,
            model="vic-clm-2.0",
            object="chat.completion.chunk"
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Final chunk with finish_reason
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
```

### 3. Dual-Path Architecture (`api/agent.py`)

Fast response first, enrichment in background:

```python
async def generate_response_with_enrichment(user_message, session_id, user_name):
    # FAST PATH: Immediate response (<2s)
    # 1. Get embedding
    embedding = await get_voyage_embedding(user_message)

    # 2. Search articles (hybrid: vector + keyword)
    results = await search_articles_hybrid(embedding, user_message)

    # 3. Generate response with Pydantic AI agent
    agent = get_fast_agent()  # groq:llama-3.3-70b-versatile
    result = await agent.run(prompt, deps=deps)
    response_text = result.data.response_text

    # ENRICHMENT PATH: Background context building
    enrichment_task = asyncio.create_task(
        run_enrichment(user_message, response_text, session_id, ...)
    )

    return response_text, enrichment_task
```

### 4. Session Context (`api/agent.py`)

In-memory storage for conversation state:

```python
@dataclass
class SessionContext:
    entities: list[ExtractedEntity]      # Extracted people, places, eras
    connections: list[EntityConnection]   # Graph connections from Zep
    suggestions: list[SuggestedTopic]     # Follow-up suggestions
    topics_discussed: list[str]           # History of topics
    last_response: str                    # For context
    last_suggested_topic: str             # For "yes" affirmation handling
    turns_since_name_used: int            # Name spacing (don't say "Dan" every turn)

# LRU store with max 100 sessions
_session_contexts: dict[str, SessionContext] = {}
```

### 5. Pydantic AI Agent (`api/agent_config.py`)

Structured output with validation:

```python
def create_fast_agent() -> Agent[VICAgentDeps, FastVICResponse]:
    return Agent(
        'groq:llama-3.3-70b-versatile',
        deps_type=VICAgentDeps,
        result_type=FastVICResponse,  # Enforces JSON schema
        system_prompt=FAST_SYSTEM_PROMPT,
        retries=2,
        model_settings={'temperature': 0.7, 'max_tokens': 300}
    )

class FastVICResponse(BaseModel):
    response_text: str
    source_titles: list[str]
```

---

## Key Patterns

### Pattern 1: Filler Phrases

Stream something immediately while searching:

```python
FILLER_PHRASES = [
    "Ah, let me think about that...",
    "Now that's an interesting question...",
    "{topic}, you say? Let me search my archives...",
]

async def stream_with_padding(user_message, session_id, user_name):
    # Start search in background
    response_task = asyncio.create_task(generate_response(...))

    # Stream filler immediately
    topic = extract_topic(user_message)
    filler = f"{topic}, you say? Let me search my archives..."
    for token in filler:
        yield create_chunk(token)
        await asyncio.sleep(0.02)  # Natural pacing

    # Wait for real response, then stream it
    response_text, _ = await response_task
    for token in response_text:
        yield create_chunk(token)
```

### Pattern 2: Affirmation Detection

Handle "yes" responses to suggestions:

```python
AFFIRMATION_WORDS = {"yes", "yeah", "sure", "okay", "please", ...}

def is_affirmation(message: str) -> tuple[bool, Optional[str]]:
    words = message.lower().split()

    # "yes" → use last suggestion
    if len(words) == 1 and words[0] in AFFIRMATION_WORDS:
        return (True, None)

    # "yeah, the Thames" → extract topic hint
    if words[0] in AFFIRMATION_WORDS and len(words) <= 3:
        topic_hint = ' '.join(words[1:])
        return (True, topic_hint)

    return (False, None)

# Store suggestions for next turn
def set_last_suggestion(session_id, topic):
    context = get_session_context(session_id)
    context.last_suggested_topic = topic
```

### Pattern 3: Hybrid Search (RRF)

Combine vector similarity with keyword matching:

```python
async def search_articles_hybrid(query_embedding, query_text, limit=5):
    # Vector search
    vector_results = await conn.fetch("""
        SELECT id, title, content,
               1 - (embedding <=> $1::vector) as similarity
        FROM articles
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """, query_embedding, limit * 2)

    # Keyword search
    keyword_results = await conn.fetch("""
        SELECT id, title, content,
               ts_rank(search_vector, query) as rank
        FROM articles, plainto_tsquery($1) query
        WHERE search_vector @@ query
        LIMIT $2
    """, query_text, limit * 2)

    # Reciprocal Rank Fusion
    scores = {}
    for rank, r in enumerate(vector_results):
        scores[r['id']] = 1 / (60 + rank)
    for rank, r in enumerate(keyword_results):
        scores[r['id']] = scores.get(r['id'], 0) + 1 / (60 + rank)

    # Return top results by RRF score
    return sorted(results, key=lambda x: scores[x['id']], reverse=True)[:limit]
```

### Pattern 4: Message Handling

Handle Hume's special messages:

```python
def extract_user_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")

            # Skip silence indicators
            if content.strip() == "[user silent]":
                return None

            # Skip greeting instructions
            if content.startswith("speak your greeting"):
                return None

            # Strip emotion tags {interested, calm}
            content = re.sub(r'\s*\{[^}]+\}\s*$', '', content)

            return content
    return None
```

### Pattern 5: Name Spacing

Don't say the user's name every turn:

```python
NAME_COOLDOWN_TURNS = 3

def should_use_name(session_id, is_greeting=False):
    context = get_session_context(session_id)

    if is_greeting and not context.name_used_in_greeting:
        return True

    if context.turns_since_name_used >= NAME_COOLDOWN_TURNS:
        return True

    return False

def mark_name_used(session_id):
    context = get_session_context(session_id)
    context.turns_since_name_used = 0
```

### Pattern 6: Content Cleaning

Remove internal structure references:

```python
def clean_section_references(text: str) -> str:
    # Remove "Section X", "Page X", etc.
    text = re.sub(r'\s*-?\s*[Ss]ection\s+\d+', '', text)
    text = re.sub(r'\s*-?\s*[Pp]age\s+\d+', '', text)

    # Fix source artifacts
    text = re.sub(r'[Yy]ou mentioned\s+', 'There was ', text)

    return text.strip()
```

---

## System Prompt Structure

```python
VIC_SYSTEM_PROMPT = """You are VIC, the voice of Vic Keegan...

## ACCURACY (NON-NEGOTIABLE)
- ONLY talk about what's IN the source material provided
- NEVER use your training knowledge

## FORBIDDEN WORDS & PHRASES
- "section", "page", "chapter" (breaks immersion)
- "you mentioned" (source artifact, not user speech)

## PERSONA
- First person: "I discovered...", "When I researched..."
- Warm, conversational British English
- 100-150 words (30-60 seconds spoken)

## RESPONSE VARIETY
Vary openings: "Ah, [topic]...", "Now, [topic]...", "Let me tell you..."

## MANDATORY FOLLOW-UP
ALWAYS end with a question about a related topic.
Only suggest topics from SAFE FOLLOW-UP TOPICS list.
"""
```

---

## Environment Variables

```bash
# LLM
GROQ_API_KEY=gsk_...

# Database
DATABASE_URL=postgres://...@...neon.tech/neondb

# Embeddings
VOYAGE_API_KEY=pa-...

# Memory
ZEP_API_KEY=z_...

# Auth (for Hume)
CLM_AUTH_TOKEN=your-secret-token
```

---

## Vercel Configuration

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

---

## Hume EVI Configuration

In the Hume console, set:
- **Custom Language Model URL**: `https://your-app.vercel.app/chat/completions`
- **Auth Header**: `Authorization: Bearer YOUR_CLM_AUTH_TOKEN`

---

## Replicating for Another App

1. **Clone the structure** - Copy the file layout
2. **Replace the domain content**:
   - Change system prompt persona
   - Update article database schema
   - Adjust entity extraction for your domain
3. **Update topic clusters** - Define safe topics for your content
4. **Adjust phonetic corrections** - Add domain-specific pronunciations
5. **Configure Hume** - Point to your CLM endpoint

---

## Performance Targets

| Path | Target | Components |
|------|--------|------------|
| Fast | <2s | Filler(200ms) + Embed(100ms) + Search(150ms) + LLM(500ms) + Stream(200ms) |
| Enriched | <5s | Fast + Graph(600ms) + Entities(50ms) + Suggestions(500ms) |

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Greeting loops | Old "speak your greeting" in history | Check MOST RECENT message only |
| Response loops on silence | Returning content for [user silent] | Return HTTP 204 No Content |
| "Yes" treated as query | No affirmation detection | Map affirmations to last suggestion |
| Section numbers in response | Source content has structure | Clean before sending to LLM |
| Name repeated every turn | No spacing logic | Track turns since name used |
