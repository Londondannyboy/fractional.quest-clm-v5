"""Tools for the Fractional Quest Pydantic AI agent - job search and career guidance."""

import re
from typing import Optional
from pydantic_ai import RunContext

from .models import JobSearchResults, JobResult, SuggestedTopic, MarketStats
from .database import search_jobs, get_job_stats, get_salary_benchmarks, get_recent_jobs
from .agent_deps import FQAgentDeps


# Phonetic corrections for voice input - business/executive terms
PHONETIC_CORRECTIONS: dict[str, str] = {
    # === EXECUTIVE TITLES (Speech recognition mishearings) ===
    "see fo": "CFO",
    "see eff oh": "CFO",
    "cee fo": "CFO",
    "c.f.o.": "CFO",
    "chief financial": "CFO",
    "chief financial officer": "CFO",

    "see em oh": "CMO",
    "cee em oh": "CMO",
    "c.m.o.": "CMO",
    "cmi": "CMO",
    "cmo": "CMO",
    "see mo": "CMO",
    "seemo": "CMO",
    "chief marketing": "CMO",
    "chief marketing officer": "CMO",
    "marketing director": "CMO",
    "vp marketing": "CMO",

    "see tee oh": "CTO",
    "cee tee oh": "CTO",
    "c.t.o.": "CTO",
    "chief technology": "CTO",
    "chief technology officer": "CTO",
    "chief tech": "CTO",

    "see oh oh": "COO",
    "cee oh oh": "COO",
    "c.o.o.": "COO",
    "chief operating": "COO",
    "chief operating officer": "COO",
    "chief ops": "COO",

    "see ro": "CRO",
    "see are oh": "CRO",
    "cee are oh": "CRO",
    "c.r.o.": "CRO",
    "chief revenue": "CRO",
    "chief revenue officer": "CRO",

    "see so": "CISO",
    "seesaw": "CISO",
    "see eye so": "CISO",
    "c.i.s.o.": "CISO",
    "chief security": "CISO",
    "chief information security": "CISO",
    "chief information security officer": "CISO",

    "see aitch are oh": "CHRO",
    "see h r o": "CHRO",
    "c.h.r.o.": "CHRO",
    "chief hr": "CHRO",
    "chief human resources": "CHRO",
    "chief human resources officer": "CHRO",
    "chief people": "CHRO",
    "chief people officer": "CHRO",

    # === FRACTIONAL WORK TERMS ===
    "fractional see fo": "fractional CFO",
    "fractional see em oh": "fractional CMO",
    "fractional see tee oh": "fractional CTO",
    "part time exec": "fractional executive",
    "part time executive": "fractional executive",
    "part-time exec": "fractional executive",
    "part-time executive": "fractional executive",
    "portfolio exec": "portfolio executive",
    "portfolio career": "portfolio career",
    "interim exec": "interim executive",
    "interim see fo": "interim CFO",

    # === INDUSTRY TERMS ===
    "saas": "SaaS",
    "sass": "SaaS",
    "b to b": "B2B",
    "b2b": "B2B",
    "be to be": "B2B",
    "b to c": "B2C",
    "b2c": "B2C",
    "be to see": "B2C",
    "fintech": "fintech",
    "fin tech": "fintech",
    "health tech": "healthtech",
    "healthtech": "healthtech",
    "ed tech": "edtech",
    "edtech": "edtech",
    "proptech": "proptech",
    "prop tech": "proptech",
    "insurtech": "insurtech",
    "insure tech": "insurtech",

    # === LOCATIONS ===
    "london": "London",
    "manchester": "Manchester",
    "birmingham": "Birmingham",
    "leeds": "Leeds",
    "bristol": "Bristol",
    "edinburgh": "Edinburgh",
    "glasgow": "Glasgow",
    "cambridge": "Cambridge",
    "oxford": "Oxford",
    "uk": "UK",
    "united kingdom": "UK",
    "remote": "remote",
    "work from home": "remote",
    "wfh": "remote",
}


def normalize_query(query: str) -> str:
    """Apply phonetic corrections to normalize the query."""
    normalized = query.lower().strip()

    for wrong, correct in PHONETIC_CORRECTIONS.items():
        # Use word boundary matching for accuracy
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
        normalized = pattern.sub(correct, normalized)

    return normalized


def extract_filters_from_query(query: str) -> dict:
    """
    Extract search filters from natural language query.

    Returns dict with:
        - search_text: The cleaned search text
        - executive_title: Detected exec title (CFO, CMO, etc.)
        - location: Detected location
        - is_remote: Whether user wants remote
    """
    query_lower = query.lower()
    filters = {
        "search_text": query,
        "executive_title": None,
        "location": None,
        "is_remote": None,
    }

    # Detect executive titles
    exec_titles = {
        "cfo": "CFO",
        "cmo": "CMO",
        "cto": "CTO",
        "coo": "COO",
        "cro": "CRO",
        "ciso": "CISO",
        "chro": "CHRO",
        "chief financial": "CFO",
        "chief marketing": "CMO",
        "chief technology": "CTO",
        "chief operating": "COO",
        "chief revenue": "CRO",
        "chief hr": "CHRO",
        "chief people": "CHRO",
    }

    for pattern, title in exec_titles.items():
        if pattern in query_lower:
            filters["executive_title"] = title
            break

    # Detect locations
    locations = ["london", "manchester", "birmingham", "leeds", "bristol",
                 "edinburgh", "glasgow", "cambridge", "oxford", "uk"]
    for loc in locations:
        if loc in query_lower:
            filters["location"] = loc.title() if loc != "uk" else "UK"
            break

    # Detect remote preference
    remote_terms = ["remote", "work from home", "wfh", "anywhere"]
    if any(term in query_lower for term in remote_terms):
        filters["is_remote"] = True

    return filters


async def search_jobs_tool(
    ctx: RunContext[FQAgentDeps],
    query: str,
) -> JobSearchResults:
    """
    Search fractional executive jobs matching the query.

    This tool searches the job database for fractional executive positions.
    Results are filtered by role type, location, and other criteria extracted from the query.

    Args:
        query: The user's job search query

    Returns:
        JobSearchResults containing matching jobs and the normalized query
    """
    import sys

    # Normalize query with phonetic corrections
    normalized_query = normalize_query(query)

    # Extract filters from query
    filters = extract_filters_from_query(normalized_query)

    print(f"[FQ Tools] Searching: '{normalized_query}' with filters: {filters}", file=sys.stderr)

    # Search database
    results = await search_jobs(
        query_text=filters["search_text"],
        executive_title=filters["executive_title"],
        location=filters["location"],
        is_remote=filters["is_remote"],
        limit=10,
    )

    jobs = [
        JobResult(
            id=r["id"],
            title=r["title"],
            company_name=r["company_name"],
            location=r.get("location"),
            city=r.get("city"),
            country=r.get("country"),
            executive_title=r.get("executive_title"),
            role_category=r.get("role_category"),
            industry=r.get("industry"),
            is_remote=r.get("is_remote"),
            estimated_hourly_rate_min=r.get("estimated_hourly_rate_min"),
            estimated_hourly_rate_max=r.get("estimated_hourly_rate_max"),
            hours_per_week=r.get("hours_per_week"),
            description_snippet=r.get("description_snippet"),
            appeal_summary=r.get("appeal_summary"),
            posted_date=str(r.get("posted_date")) if r.get("posted_date") else None,
            url=r.get("url"),
            slug=r.get("slug"),
        )
        for r in results
    ]

    return JobSearchResults(
        jobs=jobs,
        query=normalized_query,
        total_count=len(jobs),
    )


async def get_market_stats(
    ctx: RunContext[FQAgentDeps],
) -> MarketStats:
    """
    Get current job market statistics for fractional executives.

    Returns counts by role type, location, salary ranges, etc.

    Returns:
        MarketStats with job market overview
    """
    stats = await get_job_stats()

    return MarketStats(
        total_jobs=stats["total_jobs"],
        by_executive_title=stats["by_executive_title"],
        by_role_category=stats["by_role_category"],
        by_city=stats["by_city"],
        remote_jobs=stats["remote_jobs"],
        on_site_jobs=stats["on_site_jobs"],
        avg_hourly_rate_min=stats["avg_hourly_rate_min"],
        avg_hourly_rate_max=stats["avg_hourly_rate_max"],
    )


async def suggest_followup_topics(
    ctx: RunContext[FQAgentDeps],
    current_topic: str,
    mentioned_roles: list[str],
) -> list[SuggestedTopic]:
    """
    Suggest follow-up topics based on user's current search.

    Uses role types mentioned and user context to suggest relevant next steps.

    Args:
        current_topic: The main topic being discussed
        mentioned_roles: List of role types mentioned (CFO, CMO, etc.)

    Returns:
        List of suggested topics with reasons and teasers
    """
    suggestions: list[SuggestedTopic] = []
    deps = ctx.deps

    # Get user's prior interests if available
    prior_roles = deps.prior_roles if deps else []

    # Suggest related roles
    role_relations = {
        "CFO": ["COO", "CRO"],
        "CMO": ["CRO", "COO"],
        "CTO": ["CISO", "COO"],
        "COO": ["CFO", "CTO"],
        "CRO": ["CMO", "COO"],
        "CHRO": ["COO", "CFO"],
        "CISO": ["CTO", "COO"],
    }

    for role in mentioned_roles[:2]:
        related = role_relations.get(role.upper(), [])
        for related_role in related:
            if related_role not in prior_roles and related_role not in mentioned_roles:
                suggestions.append(SuggestedTopic(
                    topic=f"Fractional {related_role} roles",
                    reason=f"Often paired with {role} in similar organizations",
                    teaser=f"Companies hiring fractional {role}s often also need {related_role} support. Would you like to see those roles?",
                ))
                break

    # Suggest location or remote options if not already discussed
    if deps and not deps.preferred_remote and not deps.preferred_location:
        suggestions.append(SuggestedTopic(
            topic="Remote vs location-based roles",
            reason="Work arrangement preference",
            teaser="Are you interested in fully remote positions, or would you prefer roles in a specific city?",
        ))

    # Suggest salary benchmarks
    if len(suggestions) < 3:
        suggestions.append(SuggestedTopic(
            topic="Salary benchmarks",
            reason="Help with rate negotiation",
            teaser="Would you like to know typical day rates for fractional executives in your area?",
        ))

    return suggestions[:3]


async def get_recent_fractional_jobs(
    executive_title: Optional[str] = None,
    limit: int = 5,
) -> list[JobResult]:
    """
    Get the most recently posted fractional jobs.

    Args:
        executive_title: Optional filter by exec title
        limit: Maximum number of results

    Returns:
        List of recent jobs
    """
    results = await get_recent_jobs(
        executive_title=executive_title,
        limit=limit,
    )

    return [
        JobResult(
            id=r["id"],
            title=r["title"],
            company_name=r["company_name"],
            city=r.get("city"),
            country=r.get("country"),
            executive_title=r.get("executive_title"),
            estimated_hourly_rate_min=r.get("estimated_hourly_rate_min"),
            estimated_hourly_rate_max=r.get("estimated_hourly_rate_max"),
            hours_per_week=r.get("hours_per_week"),
            is_remote=r.get("is_remote"),
            posted_date=str(r.get("posted_date")) if r.get("posted_date") else None,
            appeal_summary=r.get("appeal_summary"),
            url=r.get("url"),
            slug=r.get("slug"),
        )
        for r in results
    ]
