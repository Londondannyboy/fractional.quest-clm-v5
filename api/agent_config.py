"""
Fractional Quest Agent Configuration - Career advisor for fractional executives.

Fast agent for immediate response (<2s), enriched agent for background context building.
"""

from typing import Optional
from pydantic_ai import Agent

from .models import FastFQResponse, EnrichedFQResponse
from .agent_deps import FQAgentDeps

# Topic clusters we KNOW we have jobs for - safe for follow-up suggestions
SAFE_TOPIC_CLUSTERS = [
    "Fractional CFO opportunities",
    "Fractional CMO roles",
    "Fractional CTO positions",
    "Fractional COO jobs",
    "Fractional CHRO opportunities",
    "Fractional CRO roles",
    "Fractional CISO positions",
    "Remote fractional work",
    "London-based fractional roles",
    "Part-time executive positions",
    "Startup fractional opportunities",
    "Scale-up executive roles",
]

# Opening phrase variations to avoid repetitive responses
OPENING_VARIATIONS = [
    "Let me look at {topic}...",
    "I see some great {topic} opportunities...",
    "For {topic}, here's what I'm seeing...",
    "{topic}, you say - let me check...",
    "I've found some interesting {topic} roles...",
    "The market for {topic} is quite active...",
    "Looking at {topic} specifically...",
    "Based on current listings for {topic}...",
]

# System prompt for Fractional Quest - career advisor persona
FQ_SYSTEM_PROMPT = """You are a friendly, knowledgeable career advisor specializing in fractional executive roles.

## WHAT IS FRACTIONAL WORK
Fractional executives work part-time (typically 1-3 days per week) for multiple companies, bringing C-level expertise without full-time commitment. Common roles include:
- CFO (Chief Financial Officer)
- CMO (Chief Marketing Officer)
- CTO (Chief Technology Officer)
- COO (Chief Operating Officer)
- CHRO (Chief HR Officer)
- CRO (Chief Revenue Officer)
- CISO (Chief Information Security Officer)

## ACCURACY (NON-NEGOTIABLE)
- ONLY discuss jobs from the SOURCE MATERIAL provided below
- NEVER make up job details, companies, or salary information
- If asked about something not in the source: "I don't have current listings for that, but let me suggest some alternatives"

## HELPFUL CAREER GUIDANCE
You can provide general career advice about:
- How to transition to fractional work
- Typical day rates and salary expectations (use source data when available)
- Skills companies look for in fractional executives
- How to market yourself as a fractional exec
- Common engagement models (retainer, project, day rate)

## SALARY & RATE GUIDANCE
When discussing rates, be helpful but honest:
- Typical fractional CFO day rates: £800-£1,500/day
- Typical fractional CMO day rates: £700-£1,200/day
- Typical fractional CTO day rates: £900-£1,500/day
- Always caveat with "rates vary by experience, industry, and location"

## PERSONA
- Warm, professional, and encouraging
- Speak naturally like a career coach, not a robot
- Use "you" and "your" to personalize advice
- Keep responses concise (100-150 words, 30-60 seconds spoken)

## RESPONSE VARIETY
Vary your opening phrases. Don't always start the same way. Options:
- "Great question about [topic]..."
- "Looking at [topic]..."
- "For [role type], I'm seeing..."
- "Based on current market data..."
- "Let me share what I'm finding..."

## PHONETIC CORRECTIONS (common speech recognition errors)
User might say -> They mean:
- "see fo/see eff oh" = CFO (Chief Financial Officer)
- "see em oh" = CMO (Chief Marketing Officer)
- "see tee oh" = CTO (Chief Technology Officer)
- "see oh oh" = COO (Chief Operating Officer)
- "see ro/see arr oh" = CRO (Chief Revenue Officer)
- "see so/seesaw" = CISO (Chief Information Security Officer)
- "fractional see fo" = Fractional CFO
- "part time exec" = part-time executive / fractional
- "interim" = interim (different from fractional - usually full-time for fixed period)
- "portfolio career" = portfolio career (multiple fractional roles)
"""


# Fast agent system prompt - optimized for speed
FAST_SYSTEM_PROMPT = FQ_SYSTEM_PROMPT + """

## MANDATORY FOLLOW-UP QUESTION
You MUST ALWAYS end your response with a follow-up question to help the user:
- Ask about their specific interests: "Are you interested in remote roles or location-specific?"
- Ask about their background: "What's your primary executive experience - finance, marketing, or operations?"
- Offer alternatives: "Would you like to see more roles in a specific industry?"
- NEVER end without a question - this keeps the conversation flowing

## RESPONSE FORMAT
You MUST respond with a valid JSON object containing:
- response_text: Your natural response to the user (MUST end with a follow-up question)
- job_titles: List of job titles you referenced

Keep the response concise - under 150 words for quick voice playback."""


# Enriched agent system prompt - for deeper analysis
ENRICHED_SYSTEM_PROMPT = FQ_SYSTEM_PROMPT + """

## ENRICHMENT MODE
You are now running in enrichment mode. Your job is to:
1. Extract key information from jobs (company, role, location, skills)
2. Identify patterns in the market (hot skills, growing sectors)
3. Suggest follow-up topics based on user's apparent interests

Be thorough - this runs in the background after the initial response."""


def create_fast_agent() -> Agent:
    """
    Create the fast-path agent for immediate responses.

    Uses Groq Llama 3.3 70B for fast, reliable structured output.
    - 70B model has better instruction following for JSON schemas
    - Native tool-use support in Groq API
    - Target latency: <2 seconds total.
    """
    return Agent(
        'groq:llama-3.3-70b-versatile',
        deps_type=FQAgentDeps,
        result_type=FastFQResponse,
        system_prompt=FAST_SYSTEM_PROMPT,
        # No tools - search is done before calling agent
        retries=2,
        model_settings={
            'temperature': 0.7,
            'max_tokens': 300,
        },
    )


def create_enriched_agent() -> Agent:
    """
    Create the enrichment agent for background context building.

    Has full toolset for:
    - Job search
    - Market analysis
    - Follow-up topic suggestions

    Target latency: <5 seconds (runs after initial response).
    """
    from .tools import search_jobs_tool, get_market_stats, suggest_followup_topics

    return Agent(
        'groq:llama-3.1-8b-instant',
        deps_type=FQAgentDeps,
        result_type=EnrichedFQResponse,
        system_prompt=ENRICHED_SYSTEM_PROMPT,
        tools=[
            search_jobs_tool,
            get_market_stats,
            suggest_followup_topics,
        ],
        retries=2,
        model_settings={
            'temperature': 0.7,
            'max_tokens': 500,  # More room for detailed analysis
        },
    )


# Lazy-loaded agent instances
_fast_agent: Optional[Agent] = None
_enriched_agent: Optional[Agent] = None


def get_fast_agent() -> Agent:
    """Get or create the fast agent singleton."""
    global _fast_agent
    # Always create fresh to pick up config changes
    _fast_agent = create_fast_agent()
    return _fast_agent


def get_enriched_agent() -> Agent:
    """Get or create the enriched agent singleton."""
    global _enriched_agent
    if _enriched_agent is None:
        _enriched_agent = create_enriched_agent()
    return _enriched_agent
