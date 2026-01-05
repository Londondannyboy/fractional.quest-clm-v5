"""
Zep Cloud Integration for Knowledge Graph Memory

Uses Zep Cloud API for:
- Jobs graph: Shared graph containing job knowledge and relationships
- Semantic search for enriching job queries
"""

import os
import httpx
from typing import Optional

ZEP_API_KEY = os.environ.get("ZEP_API_KEY", "")
ZEP_BASE_URL = "https://api.getzep.com/api/v2"
JOBS_GRAPH_ID = "fractional-jobs-graph"

# Persistent HTTP client
_zep_client: Optional[httpx.AsyncClient] = None


def get_zep_client() -> Optional[httpx.AsyncClient]:
    """Get or create persistent Zep HTTP client."""
    global _zep_client

    if not ZEP_API_KEY:
        return None

    if _zep_client is None:
        _zep_client = httpx.AsyncClient(
            base_url=ZEP_BASE_URL,
            headers={
                "Authorization": f"Api-Key {ZEP_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=10.0,
        )
    return _zep_client


async def search_jobs_graph(
    query: str,
    limit: int = 10,
) -> dict:
    """
    Search the shared jobs knowledge graph for relevant information.

    Args:
        query: Search query (e.g., "CFO with fintech experience")
        limit: Maximum results

    Returns:
        Dictionary with nodes and edges from the graph
    """
    import sys

    client = get_zep_client()
    if not client:
        print("[Zep] No API key configured", file=sys.stderr)
        return {"nodes": [], "edges": [], "error": "ZEP_API_KEY not configured"}

    try:
        # Search for nodes
        nodes_response = await client.post(
            f"/graphs/{JOBS_GRAPH_ID}/search",
            json={
                "query": query,
                "scope": "nodes",
                "limit": limit,
            }
        )
        nodes_response.raise_for_status()
        nodes_data = nodes_response.json()

        # Search for edges (relationships/facts)
        edges_response = await client.post(
            f"/graphs/{JOBS_GRAPH_ID}/search",
            json={
                "query": query,
                "scope": "edges",
                "limit": limit,
            }
        )
        edges_response.raise_for_status()
        edges_data = edges_response.json()

        print(f"[Zep] Found {len(nodes_data.get('nodes', []))} nodes, {len(edges_data.get('edges', []))} edges", file=sys.stderr)

        return {
            "nodes": nodes_data.get("nodes", []),
            "edges": edges_data.get("edges", []),
        }

    except httpx.HTTPStatusError as e:
        print(f"[Zep] HTTP error: {e.response.status_code}", file=sys.stderr)
        return {"nodes": [], "edges": [], "error": str(e)}
    except Exception as e:
        print(f"[Zep] Error: {e}", file=sys.stderr)
        return {"nodes": [], "edges": [], "error": str(e)}


async def get_user_context(
    user_id: str,
    query: str,
) -> dict:
    """
    Get user-specific context from their personal graph.

    Args:
        user_id: The user's ID
        query: What we're looking for (e.g., "skills", "preferences")

    Returns:
        Dictionary with relevant user information
    """
    import sys

    client = get_zep_client()
    if not client:
        return {"facts": [], "error": "ZEP_API_KEY not configured"}

    try:
        # Search user's graph
        response = await client.post(
            f"/users/{user_id}/graph/search",
            json={
                "query": query,
                "scope": "edges",  # Get facts/relationships
                "limit": 10,
            }
        )
        response.raise_for_status()
        data = response.json()

        # Extract facts from edges
        facts = []
        for edge in data.get("edges", []):
            if edge.get("fact"):
                facts.append(edge["fact"])

        print(f"[Zep] Found {len(facts)} facts for user {user_id}", file=sys.stderr)

        return {
            "facts": facts,
            "nodes": data.get("nodes", []),
        }

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # User not found - that's okay, they may be new
            return {"facts": [], "nodes": []}
        print(f"[Zep] HTTP error: {e.response.status_code}", file=sys.stderr)
        return {"facts": [], "error": str(e)}
    except Exception as e:
        print(f"[Zep] Error: {e}", file=sys.stderr)
        return {"facts": [], "error": str(e)}


def format_zep_context(
    graph_results: dict,
    user_context: dict = None,
) -> str:
    """
    Format Zep results into context string for LLM.

    Args:
        graph_results: Results from search_jobs_graph
        user_context: Results from get_user_context (optional)

    Returns:
        Formatted string for LLM context
    """
    parts = []

    # Add job graph context
    if graph_results.get("nodes"):
        nodes = graph_results["nodes"][:5]  # Top 5
        node_summaries = []
        for node in nodes:
            name = node.get("name", "")
            summary = node.get("summary", "")
            if name:
                node_summaries.append(f"- {name}: {summary}" if summary else f"- {name}")
        if node_summaries:
            parts.append("Related knowledge:\n" + "\n".join(node_summaries))

    if graph_results.get("edges"):
        edges = graph_results["edges"][:5]  # Top 5
        facts = [e.get("fact", "") for e in edges if e.get("fact")]
        if facts:
            parts.append("Key facts:\n" + "\n".join(f"- {f}" for f in facts))

    # Add user context
    if user_context and user_context.get("facts"):
        facts = user_context["facts"][:3]  # Top 3
        parts.append("About this user:\n" + "\n".join(f"- {f}" for f in facts))

    return "\n\n".join(parts) if parts else ""
