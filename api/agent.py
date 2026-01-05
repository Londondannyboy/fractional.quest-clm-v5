"""Pydantic AI agent for Fractional Quest - career advisor for fractional executives.

Simplified architecture for job search and career guidance.
"""

import os
import asyncio
import httpx
from typing import Optional, Tuple
from dataclasses import dataclass, field

from .models import FastFQResponse, JobResult, SuggestedTopic
from .tools import search_jobs_tool, normalize_query, extract_filters_from_query
from .database import search_jobs, get_job_stats, get_recent_jobs
from .agent_deps import FQAgentDeps
from .agent_config import get_fast_agent, FQ_SYSTEM_PROMPT, SAFE_TOPIC_CLUSTERS

# Google Gemini API for LLM
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Persistent HTTP client
_gemini_client: Optional[httpx.AsyncClient] = None


def get_gemini_client() -> httpx.AsyncClient:
    """Get or create persistent Gemini HTTP client."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            headers={
                "Content-Type": "application/json",
            },
            timeout=15.0,
        )
    return _gemini_client


# =============================================================================
# Session Context Store
# =============================================================================

@dataclass
class SessionContext:
    """Context for a conversation session."""
    roles_discussed: list[str] = field(default_factory=list)
    companies_mentioned: list[str] = field(default_factory=list)
    locations_mentioned: list[str] = field(default_factory=list)
    last_response: str = ""
    turns_since_name_used: int = 0
    name_used_in_greeting: bool = False
    last_suggested_topic: str = ""
    current_topic: str = ""
    last_interaction_time: float = 0.0
    greeted_this_session: bool = False
    preferred_location: Optional[str] = None
    preferred_remote: Optional[bool] = None


# Session store
_session_contexts: dict[str, SessionContext] = {}
MAX_SESSIONS = 100


def get_session_context(session_id: Optional[str]) -> SessionContext:
    """Get or create session context."""
    if not session_id:
        return SessionContext()

    if session_id not in _session_contexts:
        if len(_session_contexts) >= MAX_SESSIONS:
            oldest_key = next(iter(_session_contexts))
            del _session_contexts[oldest_key]
        _session_contexts[session_id] = SessionContext()

    return _session_contexts[session_id]


def update_session_context(session_id: Optional[str], context: SessionContext) -> None:
    """Update session context."""
    if session_id:
        _session_contexts[session_id] = context


# Name spacing constants
NAME_COOLDOWN_TURNS = 3


def should_use_name(session_id: Optional[str], is_greeting: bool = False) -> bool:
    """Check if we should use the user's name in this response."""
    if not session_id:
        return is_greeting

    context = get_session_context(session_id)

    if is_greeting and not context.name_used_in_greeting:
        return True

    if context.turns_since_name_used >= NAME_COOLDOWN_TURNS:
        return True

    return False


def mark_name_used(session_id: Optional[str], is_greeting: bool = False) -> None:
    """Mark that we used the name in this turn."""
    if not session_id:
        return

    context = get_session_context(session_id)
    context.turns_since_name_used = 0
    if is_greeting:
        context.name_used_in_greeting = True
    update_session_context(session_id, context)


def increment_turn_counter(session_id: Optional[str]) -> None:
    """Increment the turn counter."""
    if not session_id:
        return

    context = get_session_context(session_id)
    context.turns_since_name_used += 1
    update_session_context(session_id, context)


# Affirmation detection
AFFIRMATION_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "please",
    "absolutely", "definitely", "certainly", "alright",
}

AFFIRMATION_PHRASES = {
    "go on", "tell me more", "tell me", "go ahead", "yes please",
    "sounds good", "sounds great", "let's do it", "why not", "please do",
}


def is_affirmation(message: str) -> tuple[bool, Optional[str]]:
    """Check if a message is an affirmation/confirmation."""
    cleaned = message.lower().strip().rstrip('.!?')
    words = cleaned.split()

    if cleaned in AFFIRMATION_PHRASES:
        return (True, None)

    if len(words) == 1 and words[0] in AFFIRMATION_WORDS:
        return (True, None)

    if len(words) <= 3 and words[0] in AFFIRMATION_WORDS:
        remaining = ' '.join(words[1:])
        if remaining in AFFIRMATION_WORDS or remaining in {"then", "thanks", "please"}:
            return (True, None)
        if len(words) >= 2:
            topic_hint = ' '.join(words[1:])
            return (True, topic_hint)

    return (False, None)


def get_last_suggestion(session_id: Optional[str]) -> Optional[str]:
    """Get the last topic suggested."""
    if not session_id:
        return None
    context = get_session_context(session_id)
    return context.last_suggested_topic if context.last_suggested_topic else None


def set_last_suggestion(session_id: Optional[str], topic: str) -> None:
    """Store the topic just suggested."""
    if not session_id:
        return
    context = get_session_context(session_id)
    context.last_suggested_topic = topic
    update_session_context(session_id, context)


def set_current_topic(session_id: Optional[str], topic: str) -> None:
    """Update the current topic being discussed."""
    if not session_id:
        return
    context = get_session_context(session_id)
    context.current_topic = topic
    update_session_context(session_id, context)


def detect_topic_switch(message: str, session_id: Optional[str]) -> tuple[bool, str, str]:
    """Detect if the user is switching to a new topic."""
    if not session_id:
        return (False, "", "")

    context = get_session_context(session_id)
    current_topic = context.current_topic
    message_lower = message.lower().strip()

    # Topic switch phrases
    switch_phrases = {
        "show me", "find me", "search for", "look for",
        "what about", "how about", "let's look at",
        "switch to", "i want", "i need",
    }

    has_switch_phrase = any(phrase in message_lower for phrase in switch_phrases)

    if has_switch_phrase:
        # Extract new topic
        for phrase in switch_phrases:
            if phrase in message_lower:
                idx = message_lower.find(phrase)
                after = message[idx + len(phrase):].strip().rstrip('?.!')
                if after:
                    return (True, after, f"Let me search for {after}...")

    return (False, "", "")


def set_pending_topic_switch(session_id: Optional[str], topic: str) -> None:
    """Not used in simplified version."""
    pass


def get_pending_topic_switch(session_id: Optional[str]) -> Optional[str]:
    """Not used in simplified version."""
    return None


def clear_pending_topic_switch(session_id: Optional[str]) -> None:
    """Not used in simplified version."""
    pass


def check_returning_user(session_id: Optional[str]) -> tuple[bool, Optional[str]]:
    """Check if this is a returning user."""
    import time

    if not session_id:
        return (False, None)

    context = get_session_context(session_id)

    if context.greeted_this_session:
        return (False, None)

    if context.last_interaction_time == 0:
        return (False, None)

    current_time = time.time()
    gap = current_time - context.last_interaction_time

    # 10 minute gap = returning user
    if gap >= 600 and context.current_topic:
        return (True, context.current_topic)

    return (False, None)


def update_interaction_time(session_id: Optional[str]) -> None:
    """Update the last interaction time."""
    import time

    if not session_id:
        return

    context = get_session_context(session_id)
    context.last_interaction_time = time.time()
    update_session_context(session_id, context)


def mark_greeted_this_session(session_id: Optional[str]) -> None:
    """Mark that we've greeted the user this session."""
    if not session_id:
        return

    context = get_session_context(session_id)
    context.greeted_this_session = True
    update_session_context(session_id, context)


def set_user_emotion(session_id: Optional[str], emotion: str) -> None:
    """Store the user's detected emotion (not heavily used)."""
    pass


def get_emotion_adjustment(session_id: Optional[str]) -> str:
    """Get response adjustment based on user's emotion (not heavily used)."""
    return ""


def extract_emotion_from_message(content: str) -> tuple[str, str]:
    """Extract emotion tags from Hume message content."""
    import re

    emotion = ""
    cleaned = content

    match = re.search(r'\s*\{([^}]+)\}\s*$', content)
    if match:
        emotion = match.group(1).strip()
        cleaned = content[:match.start()].strip()

    return (cleaned, emotion)


def clean_query(query: str) -> str:
    """Clean a query by removing yes/no confirmation prefixes."""
    cleaned = query.strip()
    lower = cleaned.lower()

    prefixes = ["yes", "yeah", "yep", "sure", "okay", "ok", "no", "nah"]
    for prefix in sorted(prefixes, key=len, reverse=True):
        if lower.startswith(prefix + ","):
            cleaned = cleaned[len(prefix) + 1:].strip()
            break
        elif lower.startswith(prefix + " ") and len(cleaned) > len(prefix) + 3:
            rest = cleaned[len(prefix):].strip()
            if rest:
                cleaned = rest
                break

    return cleaned


def format_jobs_for_response(jobs: list[dict]) -> str:
    """Format job results as source material for the LLM."""
    if not jobs:
        return "No jobs found."

    formatted = []
    for job in jobs[:5]:  # Limit to 5 jobs
        rate_info = ""
        if job.get("estimated_hourly_rate_min") or job.get("estimated_hourly_rate_max"):
            rate_min = job.get("estimated_hourly_rate_min", "?")
            rate_max = job.get("estimated_hourly_rate_max", "?")
            rate_info = f" (approx. £{rate_min}-£{rate_max}/hr)"

        location = job.get("city") or job.get("location") or "Location not specified"
        if job.get("is_remote"):
            location = "Remote"

        formatted.append(f"""
**{job.get('title', 'Untitled')}** at {job.get('company_name', 'Unknown Company')}
- Location: {location}
- Type: {job.get('executive_title', 'Executive Role')}{rate_info}
- Hours: {job.get('hours_per_week', 'Not specified')}
- Appeal: {job.get('appeal_summary', job.get('description_snippet', 'No description'))[:200]}
""")

    return "\n".join(formatted)


async def generate_response(
    user_message: str,
    session_id: Optional[str] = None,
    user_name: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Generate a response to the user's message.

    Args:
        user_message: The user's question
        session_id: Optional session ID for context
        user_name: Optional user's first name
        user_id: Optional user ID for Zep context
    """
    import sys
    from .zep_client import search_jobs_graph, get_user_context, format_zep_context

    try:
        # Normalize query
        normalized_query = normalize_query(user_message)
        filters = extract_filters_from_query(normalized_query)

        print(f"[FQ Agent] Query: '{user_message}' -> Normalized: '{normalized_query}'", file=sys.stderr)
        print(f"[FQ Agent] Filters: {filters}", file=sys.stderr)

        # Search for jobs in database
        results = await search_jobs(
            query_text=filters["search_text"],
            executive_title=filters["executive_title"],
            location=filters["location"],
            is_remote=filters["is_remote"],
            limit=5,
        )

        print(f"[FQ Agent] Found {len(results)} jobs from database", file=sys.stderr)

        # Enrich with Zep knowledge graph (parallel)
        zep_context = ""
        try:
            graph_results = await search_jobs_graph(normalized_query, limit=5)
            user_context = None
            if user_id:
                user_context = await get_user_context(user_id, normalized_query)
            zep_context = format_zep_context(graph_results, user_context)
            if zep_context:
                print(f"[FQ Agent] Got Zep context: {len(zep_context)} chars", file=sys.stderr)
        except Exception as e:
            print(f"[FQ Agent] Zep enrichment failed (continuing): {e}", file=sys.stderr)

        if not results:
            # Get stats and suggest alternatives
            stats = await get_job_stats()
            top_titles = stats.get("by_executive_title", [])[:3]
            suggestions = ", ".join([t["title"] for t in top_titles]) if top_titles else "CFO, CMO, CTO"

            return (
                f"I don't currently have any listings matching that search. "
                f"We currently have {stats.get('total_jobs', 0)} active fractional roles, "
                f"mostly in {suggestions}. "
                f"Would you like to see roles in one of those areas?"
            )

        # Format jobs as source material
        source_content = format_jobs_for_response(results)
        job_titles = [r.get("title", "") for r in results]

        # Build name instruction
        name_instruction = ""
        if user_name:
            name_instruction = f"\n\nThe user's name is {user_name}. Address them naturally."
        else:
            name_instruction = "\n\nYou don't know the user's name."

        # Build prompt with Zep context if available
        zep_section = ""
        if zep_context:
            zep_section = f"""
ADDITIONAL CONTEXT (from knowledge graph):
{zep_context}
"""

        prompt = f"""Question: "{user_message}"
{name_instruction}

AVAILABLE JOBS (use only this data):
{source_content}
{zep_section}
Respond naturally as a helpful career advisor. Keep it conversational and under 150 words.
Mention 1-2 specific jobs that match their query.
End with a follow-up question about their preferences (location, remote work, specific industry, etc.)."""

        # Call Google Gemini LLM
        client = get_gemini_client()

        # Gemini uses a different API format
        llm_response = await client.post(
            f"/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": f"{FQ_SYSTEM_PROMPT}\n\n{prompt}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 300,
                    "temperature": 0.7,
                }
            },
        )
        llm_response.raise_for_status()
        data = llm_response.json()
        response_text = data["candidates"][0]["content"]["parts"][0]["text"]

        return response_text.strip()

    except Exception as e:
        import traceback
        error_type = type(e).__name__
        print(f"[FQ Agent Error] {error_type}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        return (
            f"I'm having trouble searching right now ({error_type}). "
            "Could you try rephrasing your question?"
        )


async def generate_response_with_enrichment(
    user_message: str,
    session_id: Optional[str] = None,
    user_name: Optional[str] = None,
) -> tuple[str, Optional[asyncio.Task]]:
    """
    Generate response (enrichment not implemented for simplicity).

    Returns:
        Tuple of (response_text, None)
    """
    response = await generate_response(user_message, session_id, user_name)
    return response, None


def get_suggestion_teaser(session_id: Optional[str]) -> Optional[str]:
    """Get a follow-up suggestion teaser."""
    # Not implemented in simplified version
    return None


def get_popular_topics(limit: int = 10) -> list[tuple[str, int]]:
    """Get popular topics (not tracked in simplified version)."""
    return []
