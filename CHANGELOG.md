# VIC CLM Changelog & Feature Tracker

A comprehensive log of all features implemented and planned for VIC - the voice of Vic Keegan.

---

## Current Version: 2.1.0 (December 2024)

### Deployed: https://vic-clm.vercel.app

---

## Implemented Features

### Core Architecture

| Feature | Status | File(s) | Description |
|---------|--------|---------|-------------|
| Dual-path architecture | ✅ Done | `agent.py` | Fast response + background enrichment |
| Pydantic AI agents | ✅ Done | `agent_config.py` | Structured output with schema validation |
| Groq LLM (Llama 3.3 70B) | ✅ Done | `agent_config.py` | Fast inference for responses |
| Voyage AI embeddings | ✅ Done | `tools.py` | Vector search for articles |
| Neon PostgreSQL | ✅ Done | `database.py` | Article storage with pgvector |
| Zep memory | ✅ Done | `agent.py` | Conversation history & knowledge graph |
| Hybrid search (RRF) | ✅ Done | `database.py` | Vector + keyword with reciprocal rank fusion |

### Conversation Flow

| Feature | Status | File(s) | Description |
|---------|--------|---------|-------------|
| Streaming SSE responses | ✅ Done | `index.py` | OpenAI-compatible streaming for Hume EVI |
| Filler phrases | ✅ Done | `index.py` | "Let me think..." while searching |
| Affirmation detection | ✅ Done | `agent.py` | "Yes" → use last suggestion |
| Topic switch detection | ✅ Done | `agent.py` | Detects when user changes topic |
| Returning user detection | ✅ Done | `agent.py` | "Welcome back, last time we discussed..." |
| Name spacing | ✅ Done | `agent.py` | Don't say name every turn (3-turn cooldown) |
| Session context store | ✅ Done | `agent.py` | In-memory LRU cache (100 sessions) |

### Voice & Speech

| Feature | Status | File(s) | Description |
|---------|--------|---------|-------------|
| Phonetic corrections | ✅ Done | `tools.py` | 80+ corrections (tie burn → tyburn) |
| Emotion-aware responses | ✅ Done | `agent.py` | Adjust tone based on Hume emotion tags |
| Response pacing | ✅ Done | `index.py` | Micro-delays at punctuation for natural speech |
| Voice correction capture | ✅ Done | `agent.py` | Store user corrections for review |

### Response Quality

| Feature | Status | File(s) | Description |
|---------|--------|---------|-------------|
| Confidence indicators | ✅ Done | `agent.py` | Soften responses when search scores are low |
| "I'm not sure" responses | ✅ Done | `agent.py` | Graceful uncertainty handling |
| Section reference cleaning | ✅ Done | `agent.py` | Remove "Section 16", "you mentioned" |
| Post-validation | ✅ Done | `agent.py` | Catch hallucinated dates/architects |
| Safe topic clusters | ✅ Done | `agent_config.py` | Only suggest topics we have content on |

### Performance

| Feature | Status | File(s) | Description |
|---------|--------|---------|-------------|
| Embedding cache | ✅ Done | `tools.py` | 10-min TTL, 200 entries max |
| Persistent HTTP clients | ✅ Done | `agent.py`, `tools.py` | Connection reuse for Groq/Voyage/Zep |
| Popular topics tracking | ✅ Done | `agent.py` | Track query frequency with time decay |

### Debug Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Health check & version info |
| `GET /health` | Simple health check |
| `GET /debug/last-request` | Last request received |
| `GET /debug/session/{id}` | Session context (entities, connections, suggestions) |
| `GET /debug/search` | Test search functionality |
| `GET /debug/popular-topics` | View trending topics |
| `GET /debug/cache-stats` | Embedding cache statistics |

---

## Configuration Constants

### Thresholds (`agent.py`)

```python
HIGH_CONFIDENCE_THRESHOLD = 0.025   # Very relevant results
MEDIUM_CONFIDENCE_THRESHOLD = 0.015  # Decent match
LOW_CONFIDENCE_THRESHOLD = 0.01      # Weak match
NAME_COOLDOWN_TURNS = 3              # Turns before using name again
RETURNING_USER_GAP_SECONDS = 600     # 10 minutes = "returning" user
MAX_SESSIONS = 100                   # LRU session cache size
```

### Cache Settings (`tools.py`)

```python
EMBEDDING_CACHE_TTL_SECONDS = 600    # 10 minutes
MAX_EMBEDDING_CACHE_SIZE = 200       # Max cached embeddings
```

### Topic Tracking (`agent.py`)

```python
MAX_TRACKED_TOPICS = 500             # Max topics to track
TOPIC_DECAY_HOURS = 24               # Decay weighting for popularity
```

---

## System Prompts

### Key Rules in `agent_config.py`

1. **Accuracy**: Only use source material, never training knowledge
2. **Answer the question**: First sentence must address their actual question
3. **Stay on topic**: Brief connections allowed, but focus on what was asked
4. **Forbidden words**: "section", "page", "chapter", "you mentioned"
5. **Response variety**: Vary opening phrases
6. **Mandatory follow-up**: Always end with a related question

---

## Phonetic Corrections (Expanded)

Categories in `tools.py`:
- **Names**: ignacio→ignatius, peeps→pepys, brunell→brunel
- **Thorney Island**: thorny, fawny, tourney, tawny
- **Tyburn**: tie burn, tieburn, tyler burn, tybourne
- **Neighborhoods**: vox hall→vauxhall, south work→southwark, green witch→greenwich
- **Landmarks**: trafalger→trafalgar, westminister→westminster
- **Rivers**: tems→thames, west bourne→westbourne
- **Eras**: victorean→victorian, medival→medieval
- **Venues**: alambra→alhambra, hipodrome→hippodrome

---

## Emotion Adjustments

When Hume detects user emotion, VIC adjusts:

| Emotion | Adjustment |
|---------|------------|
| confused | Explain simply, short sentences |
| interested | More detail, fascinating facts |
| excited | Match energy, be enthusiastic |
| bored | Keep brief, offer topic switch |
| contemplative | Thoughtful questions, give space |
| curious | Encourage, offer to go deeper |
| skeptical | Precise facts, cite sources |

---

## Future Improvements (Ideas)

### High Priority
- [ ] Persist popular topics to database (currently in-memory)
- [ ] Admin dashboard for voice corrections review
- [ ] A/B test response styles

### Medium Priority
- [ ] Multi-turn conversation summary
- [ ] User preference learning (favorite topics)
- [ ] Proactive suggestions after silence

### Low Priority
- [ ] Voice activity detection improvements
- [ ] Multiple voice personas
- [ ] Integration with lost.london frontend

---

## Version History

### v2.1.0 (December 2024)
- Added emotion-aware responses
- Added response pacing (punctuation delays)
- Added confidence indicators
- Added popular topics tracking
- Expanded phonetic corrections (80+)
- Added uncertainty responses
- Added embedding cache
- Added debug endpoints for cache/topics

### v2.0.0 (December 2024)
- Migrated to Pydantic AI agents
- Dual-path architecture (fast + enrichment)
- Groq LLM integration (Llama 3.3 70B)
- Zep memory integration
- Affirmation detection
- Topic switch detection
- Returning user detection
- Name spacing

### v1.0.0 (November 2024)
- Initial CLM implementation
- Basic article search
- Streaming responses
- Hume EVI integration

---

## Environment Variables

```bash
GROQ_API_KEY=gsk_...          # Groq LLM API
DATABASE_URL=postgres://...    # Neon PostgreSQL
VOYAGE_API_KEY=pa-...          # Voyage AI embeddings
ZEP_API_KEY=z_...              # Zep memory
CLM_AUTH_TOKEN=...             # Hume authentication
```

---

## Testing

### Quick Test Commands

```bash
# Health check
curl https://vic-clm.vercel.app/health

# Test query
curl -X POST https://vic-clm.vercel.app/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me about the Royal Aquarium"}]}'

# Check cache
curl https://vic-clm.vercel.app/debug/cache-stats

# Check popular topics
curl https://vic-clm.vercel.app/debug/popular-topics

# Check session context
curl https://vic-clm.vercel.app/debug/session/YOUR_SESSION_ID
```

---

## Files Overview

```
vic-clm/
├── api/
│   ├── index.py          # FastAPI app, /chat/completions, streaming
│   ├── agent.py          # Response generation, session context, helpers
│   ├── agent_config.py   # System prompts, Pydantic AI agent setup
│   ├── agent_deps.py     # Agent dependencies dataclass
│   ├── models.py         # Pydantic models (requests/responses)
│   ├── tools.py          # Search, embeddings, phonetic corrections
│   └── database.py       # Neon PostgreSQL, hybrid search
├── vercel.json           # Vercel deployment config
├── requirements.txt      # Python dependencies
├── ARCHITECTURE.md       # Technical architecture guide
└── CHANGELOG.md          # This file
```

---

*Last updated: December 27, 2024*
