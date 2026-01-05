"""
Fractional Quest CLM Server - Custom Language Model for Hume EVI

This FastAPI server implements the OpenAI-compatible /chat/completions endpoint
that Hume EVI requires for Custom Language Model integration.

Career advisor for fractional executive roles.
"""

import os
import time
import json
import random
import asyncio
import re
from uuid import uuid4
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Security, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
import tiktoken

from .agent import (
    generate_response,
    generate_response_with_enrichment,
    get_suggestion_teaser,
    should_use_name,
    mark_name_used,
    increment_turn_counter,
    is_affirmation,
    get_last_suggestion,
    set_last_suggestion,
    detect_topic_switch,
    set_current_topic,
    set_pending_topic_switch,
    get_pending_topic_switch,
    clear_pending_topic_switch,
    clean_query,
    check_returning_user,
    update_interaction_time,
    mark_greeted_this_session,
    set_user_emotion,
    get_emotion_adjustment,
    extract_emotion_from_message,
)
from .database import Database

# Token for authenticating Hume requests
CLM_AUTH_TOKEN = os.environ.get("CLM_AUTH_TOKEN", "")

# Tokenizer for streaming response chunks
enc = tiktoken.encoding_for_model("gpt-4o")

# Filler phrases while searching (career advisor style)
FILLER_PHRASES = [
    "Let me check our current listings...",
    "I'm looking through the fractional roles we have...",
    "Give me a moment to find the best matches...",
    "Let me see what's available...",
    "Searching our job database now...",
]

# Topic-aware filler phrases
TOPIC_FILLER_PHRASES = [
    "Looking for {topic} roles...",
    "{topic} positions, let me check...",
    "Searching for {topic} opportunities...",
]


def extract_topic(user_message: str) -> Optional[str]:
    """Extract a potential topic/subject from the user's message."""
    # Look for executive role mentions
    role_patterns = [
        r'\b(cfo|cmo|cto|coo|cro|ciso|chro)\b',
        r'\b(chief\s+(?:financial|marketing|technology|operating|revenue|security|hr))\b',
        r'\b(fractional\s+\w+)\b',
    ]

    message_lower = user_message.lower()
    for pattern in role_patterns:
        match = re.search(pattern, message_lower)
        if match:
            return match.group(1).upper() if len(match.group(1)) <= 4 else match.group(1).title()

    # Check for location mentions
    locations = ["london", "manchester", "remote", "birmingham", "leeds"]
    for loc in locations:
        if loc in message_lower:
            return loc.title()

    return None


# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage database connection lifecycle."""
    # Startup
    yield
    # Shutdown - close database pool
    await Database.close()


app = FastAPI(
    title="Fractional Quest CLM",
    description="Custom Language Model for Fractional Quest voice assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
    """Verify the Bearer token from Hume."""
    if not CLM_AUTH_TOKEN:
        # No token configured - allow all requests (dev mode)
        return True
    if not credentials:
        return False
    return credentials.credentials == CLM_AUTH_TOKEN


def extract_user_message_and_emotion(messages: list[dict]) -> tuple[Optional[str], str]:
    """Extract the last user message and emotion from conversation history."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                ]
                content = " ".join(text_parts)

            if content and content.lower().strip() == "[user silent]":
                return (None, "")

            if content and content.lower().startswith("speak your greeting"):
                return (None, "")

            emotion = ""
            if content and "{" in content:
                cleaned, emotion = extract_emotion_from_message(content)
                content = cleaned

            return (content, emotion)
    return (None, "")


def extract_user_name_from_messages(messages: list[dict]) -> Optional[str]:
    """Extract user name from system message if present."""
    import sys

    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                match = re.search(r"USER'S NAME:\s*(\w+)", content, re.IGNORECASE)
                if match:
                    return match.group(1)
                match = re.search(r"(?:Hello|Welcome back),?\s+(\w+)", content)
                if match:
                    return match.group(1)
    return None


def extract_session_id(request: Request, body: Optional[dict] = None) -> Optional[str]:
    """Extract custom_session_id from request."""
    session_id = request.query_params.get("custom_session_id")
    if session_id:
        return session_id

    for header_name in ["x-hume-session-id", "x-session-id", "x-custom-session-id"]:
        session_id = request.headers.get(header_name)
        if session_id:
            return session_id

    if body:
        session_id = body.get("custom_session_id") or body.get("session_id")
        if session_id:
            return session_id
        metadata = body.get("metadata", {})
        session_id = metadata.get("custom_session_id") or metadata.get("session_id")
        if session_id:
            return session_id

    return None


def extract_user_name_from_session(session_id: Optional[str]) -> Optional[str]:
    """Extract user_name from session ID."""
    if not session_id:
        return None
    if '|' in session_id:
        name = session_id.split('|')[0]
        if name and len(name) >= 2 and len(name) <= 20 and name.isalpha():
            return name
    return None


def create_chunk(chunk_id: str, created: int, content: str, session_id: Optional[str], is_first: bool = False) -> str:
    """Create a single SSE chunk in OpenAI format."""
    chunk = ChatCompletionChunk(
        id=chunk_id,
        choices=[
            Choice(
                delta=ChoiceDelta(
                    content=content,
                    role="assistant" if is_first else None,
                ),
                finish_reason=None,
                index=0,
            )
        ],
        created=created,
        model="fractional-quest-clm-1.0",
        object="chat.completion.chunk",
        system_fingerprint=session_id,
    )
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


async def stream_response(text: str, session_id: Optional[str] = None):
    """Stream response as OpenAI-compatible ChatCompletionChunks."""
    chunk_id = str(uuid4())
    created = int(time.time())

    tokens = enc.encode(text)

    for i, token_id in enumerate(tokens):
        token_text = enc.decode([token_id])
        yield create_chunk(chunk_id, created, token_text, session_id, is_first=(i == 0))

        if token_text.rstrip() in {'.', '!', '?'}:
            await asyncio.sleep(0.05)
        elif token_text.rstrip() in {',', ';', ':', '...'}:
            await asyncio.sleep(0.02)

    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        choices=[
            Choice(
                delta=ChoiceDelta(),
                finish_reason="stop",
                index=0,
            )
        ],
        created=created,
        model="fractional-quest-clm-1.0",
        object="chat.completion.chunk",
        system_fingerprint=session_id,
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"


async def stream_with_padding(
    user_message: str,
    session_id: Optional[str] = None,
    user_name: Optional[str] = None
):
    """Stream filler phrases while generating the real response."""
    chunk_id = str(uuid4())
    created = int(time.time())

    # Start generating response in background
    response_task = asyncio.create_task(
        generate_response_with_enrichment(user_message, session_id, user_name)
    )

    # Choose a filler phrase
    topic = extract_topic(user_message)
    if topic and random.random() > 0.3:
        filler = random.choice(TOPIC_FILLER_PHRASES).format(topic=topic)
    else:
        filler = random.choice(FILLER_PHRASES)

    # Stream the filler phrase
    filler_tokens = enc.encode(filler)
    for i, token_id in enumerate(filler_tokens):
        token_text = enc.decode([token_id])
        yield create_chunk(chunk_id, created, token_text, session_id, is_first=(i == 0))
        await asyncio.sleep(0.02)

    yield create_chunk(chunk_id, created, " ", session_id)

    # Wait for response
    response_text, enrichment_task = await response_task

    # Stream the actual response
    response_tokens = enc.encode(response_text)
    for token_id in response_tokens:
        token_text = enc.decode([token_id])
        yield create_chunk(chunk_id, created, token_text, session_id)

    # Final chunk
    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        choices=[
            Choice(
                delta=ChoiceDelta(),
                finish_reason="stop",
                index=0,
            )
        ],
        created=created,
        model="fractional-quest-clm-1.0",
        object="chat.completion.chunk",
        system_fingerprint=session_id,
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat/completions")
async def chat_completions(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    """OpenAI-compatible chat completions endpoint for Hume CLM."""
    global _last_request_debug

    if not verify_token(credentials):
        raise HTTPException(status_code=401, detail="Invalid or missing auth token")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages", [])

    session_id = extract_session_id(request, body)
    user_message_extracted, user_emotion = extract_user_message_and_emotion(messages)
    topic_extracted = extract_topic(user_message_extracted) if user_message_extracted else None

    if user_emotion and session_id:
        set_user_emotion(session_id, user_emotion)

    # Store for debugging
    _last_request_debug = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "messages_count": len(messages),
        "user_message_extracted": user_message_extracted,
        "topic_extracted": topic_extracted,
    }

    user_name = extract_user_name_from_session(session_id)
    if not user_name:
        user_name = extract_user_name_from_messages(messages)

    import sys
    print(f"[FQ CLM] Session: {session_id}, User: {user_name}, Message: {user_message_extracted}", file=sys.stderr)

    user_message = user_message_extracted

    # Check for greeting request
    most_recent_user_content = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                most_recent_user_content = content.lower().strip()
            break

    is_greeting_request = (
        most_recent_user_content is not None and
        most_recent_user_content.startswith("speak your greeting")
    )

    if is_greeting_request:
        is_returning, last_topic = check_returning_user(session_id)

        if is_returning and user_name and last_topic:
            greeting = f"Welcome back, {user_name}. Last time we were discussing {last_topic}. Would you like to continue with that, or explore something different?"
            mark_name_used(session_id, is_greeting=True)
        elif user_name:
            greeting = f"Hello {user_name}! I'm your fractional career advisor. I can help you find part-time executive roles like CFO, CMO, or CTO positions. What kind of role are you looking for?"
            mark_name_used(session_id, is_greeting=True)
        else:
            greeting = "Hello! I'm your fractional career advisor. I can help you discover part-time executive opportunities. Are you looking for a specific role like CFO, CMO, or CTO?"

        mark_greeted_this_session(session_id)
        update_interaction_time(session_id)

        return StreamingResponse(
            stream_response(greeting, session_id),
            media_type="text/event-stream",
        )

    if not user_message:
        import sys
        print(f"[FQ CLM] No user message (silence), returning 204", file=sys.stderr)
        return Response(status_code=204)

    # Handle affirmations
    actual_query = user_message
    is_affirm, topic_hint = is_affirmation(user_message)

    if is_affirm:
        if topic_hint:
            actual_query = topic_hint
        else:
            last_suggestion = get_last_suggestion(session_id)
            if last_suggestion:
                actual_query = last_suggestion

    # Check for topic switch
    is_switch, new_topic, _ = detect_topic_switch(user_message, session_id)
    if is_switch:
        set_current_topic(session_id, new_topic)
        actual_query = new_topic

    actual_query = clean_query(actual_query)

    # Check name usage
    use_name = should_use_name(session_id, is_greeting=False)
    effective_name = user_name if use_name else None

    if use_name and user_name:
        mark_name_used(session_id)

    increment_turn_counter(session_id)
    update_interaction_time(session_id)

    return StreamingResponse(
        stream_with_padding(actual_query, session_id, effective_name),
        media_type="text/event-stream",
    )


@app.get("/")
async def root():
    """Health check and info endpoint."""
    return {
        "status": "ok",
        "service": "Fractional Quest CLM",
        "version": "1.0.0",
        "description": "Custom Language Model for Fractional Quest voice assistant",
        "endpoint": "/chat/completions",
    }


@app.get("/health")
async def health():
    """Health check for monitoring."""
    return {"status": "healthy"}


@app.get("/debug/last-request")
async def debug_last_request():
    """Return the last request received for debugging."""
    return _last_request_debug


@app.get("/debug/search")
async def debug_search():
    """Debug endpoint to test job search functionality."""
    from .database import search_jobs, get_job_stats

    try:
        # Test search
        results = await search_jobs(query_text="CFO", limit=3)
        stats = await get_job_stats()

        return {
            "status": "ok",
            "total_jobs": stats.get("total_jobs", 0),
            "search_results": len(results),
            "sample_jobs": [
                {"title": r.get("title"), "company": r.get("company_name")}
                for r in results[:3]
            ],
            "by_title": stats.get("by_executive_title", [])[:5],
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# Store last request for debugging
_last_request_debug: dict = {"status": "no requests yet"}


# For local development with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
