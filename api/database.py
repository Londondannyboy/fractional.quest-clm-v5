"""Database connection and queries for Neon PostgreSQL - Fractional Jobs."""

import os
import asyncpg
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

DATABASE_URL = os.environ.get("DATABASE_URL", "")


class Database:
    """Async database connection manager for Neon PostgreSQL."""

    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        """Get or create connection pool."""
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        return cls._pool

    @classmethod
    async def close(cls) -> None:
        """Close the connection pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None


@asynccontextmanager
async def get_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Get a database connection from the pool."""
    pool = await Database.get_pool()
    async with pool.acquire() as conn:
        yield conn


async def search_jobs(
    query_text: str,
    executive_title: Optional[str] = None,
    location: Optional[str] = None,
    is_remote: Optional[bool] = None,
    limit: int = 10,
) -> list[dict]:
    """
    Search fractional executive jobs with filters.

    Args:
        query_text: Search text for title/description
        executive_title: Filter by exec title (CFO, CMO, etc.)
        location: Filter by city or country
        is_remote: Filter for remote jobs
        limit: Maximum number of results

    Returns:
        List of matching jobs
    """
    import sys

    async with get_connection() as conn:
        # Base query - always return fractional jobs
        query = """
            SELECT
                id::text,
                title,
                normalized_title,
                company_name,
                location,
                city::text,
                country,
                employment_type,
                workplace_type,
                is_remote,
                executive_title::text,
                role_category::text,
                industry::text,
                salary_min,
                salary_max,
                salary_currency,
                estimated_hourly_rate_min,
                estimated_hourly_rate_max,
                hours_per_week,
                description_snippet,
                full_description,
                appeal_summary,
                key_deliverables,
                skills_required,
                requirements,
                posted_date,
                url,
                slug
            FROM jobs
            WHERE is_active = true
              AND is_fractional = true
        """
        params = []
        param_idx = 1
        has_filter = False

        # If executive_title is provided, filter by it OR search in title
        if executive_title:
            query += f" AND (executive_title::text ILIKE ${param_idx} OR title ILIKE ${param_idx})"
            params.append(f"%{executive_title}%")
            param_idx += 1
            has_filter = True

        # If location provided, filter by location
        if location:
            query += f" AND (city::text ILIKE ${param_idx} OR country ILIKE ${param_idx} OR location ILIKE ${param_idx})"
            params.append(f"%{location}%")
            param_idx += 1
            has_filter = True

        if is_remote is not None:
            query += f" AND is_remote = ${param_idx}"
            params.append(is_remote)
            param_idx += 1
            has_filter = True

        # If no specific filters but query has keywords, do a broad text search
        if not has_filter and query_text:
            # Clean query text - remove common filler words
            clean_words = [w for w in query_text.lower().split()
                          if w not in ('show', 'me', 'find', 'get', 'what', 'are', 'the', 'jobs', 'roles', 'positions', 'available', 'have', 'you', 'do', 'any')]
            if clean_words:
                search_term = ' '.join(clean_words[:3])  # Use first 3 meaningful words
                query += f""" AND (
                    title ILIKE ${param_idx}
                    OR normalized_title ILIKE ${param_idx}
                    OR description_snippet ILIKE ${param_idx}
                    OR company_name ILIKE ${param_idx}
                    OR executive_title::text ILIKE ${param_idx}
                )"""
                params.append(f"%{search_term}%")
                param_idx += 1

        query += f" ORDER BY posted_date DESC NULLS LAST LIMIT ${param_idx}"
        params.append(limit)

        print(f"[FQ Search] Filters: exec_title={executive_title}, loc={location}, remote={is_remote}", file=sys.stderr)
        print(f"[FQ Search] Query params: {params}", file=sys.stderr)

        results = await conn.fetch(query, *params)

        print(f"[FQ Search] Found {len(results)} jobs", file=sys.stderr)

        return [dict(r) for r in results]


async def get_job_stats() -> dict:
    """
    Get statistics about available fractional jobs.

    Returns:
        Dictionary with job counts by category, location, etc.
    """
    async with get_connection() as conn:
        # Total active fractional jobs
        total = await conn.fetchval("""
            SELECT COUNT(*) FROM jobs
            WHERE is_active = true AND is_fractional = true
        """)

        # By executive title
        by_title = await conn.fetch("""
            SELECT executive_title::text as title, COUNT(*) as count
            FROM jobs
            WHERE is_active = true AND is_fractional = true
            AND executive_title IS NOT NULL
            GROUP BY executive_title
            ORDER BY count DESC
        """)

        # By role category
        by_category = await conn.fetch("""
            SELECT role_category::text as category, COUNT(*) as count
            FROM jobs
            WHERE is_active = true AND is_fractional = true
            AND role_category IS NOT NULL
            GROUP BY role_category
            ORDER BY count DESC
        """)

        # By location (city)
        by_city = await conn.fetch("""
            SELECT city::text as city, COUNT(*) as count
            FROM jobs
            WHERE is_active = true AND is_fractional = true
            AND city IS NOT NULL
            GROUP BY city
            ORDER BY count DESC
            LIMIT 10
        """)

        # Remote vs on-site
        remote_count = await conn.fetchval("""
            SELECT COUNT(*) FROM jobs
            WHERE is_active = true AND is_fractional = true AND is_remote = true
        """)

        # Average salary ranges
        salary_stats = await conn.fetchrow("""
            SELECT
                AVG(salary_min) as avg_min,
                AVG(salary_max) as avg_max,
                AVG(estimated_hourly_rate_min) as avg_hourly_min,
                AVG(estimated_hourly_rate_max) as avg_hourly_max
            FROM jobs
            WHERE is_active = true AND is_fractional = true
            AND (salary_min IS NOT NULL OR estimated_hourly_rate_min IS NOT NULL)
        """)

        return {
            "total_jobs": total,
            "by_executive_title": [{"title": r["title"], "count": r["count"]} for r in by_title],
            "by_role_category": [{"category": r["category"], "count": r["count"]} for r in by_category],
            "by_city": [{"city": r["city"], "count": r["count"]} for r in by_city],
            "remote_jobs": remote_count,
            "on_site_jobs": total - remote_count if total else 0,
            "avg_salary_min": float(salary_stats["avg_min"]) if salary_stats and salary_stats["avg_min"] else None,
            "avg_salary_max": float(salary_stats["avg_max"]) if salary_stats and salary_stats["avg_max"] else None,
            "avg_hourly_rate_min": float(salary_stats["avg_hourly_min"]) if salary_stats and salary_stats["avg_hourly_min"] else None,
            "avg_hourly_rate_max": float(salary_stats["avg_hourly_max"]) if salary_stats and salary_stats["avg_hourly_max"] else None,
        }


async def get_salary_benchmarks(
    executive_title: Optional[str] = None,
    location: Optional[str] = None,
) -> dict:
    """
    Get salary benchmarks for fractional executive roles.

    Args:
        executive_title: Filter by exec title (CFO, CMO, etc.)
        location: Filter by city or country

    Returns:
        Dictionary with salary ranges and percentiles
    """
    async with get_connection() as conn:
        query = """
            SELECT
                executive_title::text as title,
                city::text as city,
                country,
                AVG(estimated_hourly_rate_min) as avg_hourly_min,
                AVG(estimated_hourly_rate_max) as avg_hourly_max,
                MIN(estimated_hourly_rate_min) as min_hourly,
                MAX(estimated_hourly_rate_max) as max_hourly,
                AVG(salary_min) as avg_salary_min,
                AVG(salary_max) as avg_salary_max,
                COUNT(*) as job_count
            FROM jobs
            WHERE is_active = true AND is_fractional = true
        """
        params = []
        param_idx = 1

        if executive_title:
            query += f" AND executive_title::text ILIKE ${param_idx}"
            params.append(f"%{executive_title}%")
            param_idx += 1

        if location:
            query += f" AND (city::text ILIKE ${param_idx} OR country ILIKE ${param_idx})"
            params.append(f"%{location}%")
            param_idx += 1

        query += " GROUP BY executive_title, city, country ORDER BY job_count DESC LIMIT 20"

        results = await conn.fetch(query, *params)

        return {
            "benchmarks": [
                {
                    "title": r["title"],
                    "city": r["city"],
                    "country": r["country"],
                    "avg_hourly_min": float(r["avg_hourly_min"]) if r["avg_hourly_min"] else None,
                    "avg_hourly_max": float(r["avg_hourly_max"]) if r["avg_hourly_max"] else None,
                    "hourly_range": f"£{int(r['min_hourly'] or 0)}-£{int(r['max_hourly'] or 0)}/hr",
                    "avg_salary_min": float(r["avg_salary_min"]) if r["avg_salary_min"] else None,
                    "avg_salary_max": float(r["avg_salary_max"]) if r["avg_salary_max"] else None,
                    "job_count": r["job_count"],
                }
                for r in results
            ]
        }


async def get_recent_jobs(
    executive_title: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Get the most recently posted fractional jobs.

    Args:
        executive_title: Filter by exec title (CFO, CMO, etc.)
        limit: Maximum number of results

    Returns:
        List of recent jobs with key details
    """
    async with get_connection() as conn:
        query = """
            SELECT
                id::text,
                title,
                normalized_title,
                company_name,
                city::text,
                country,
                executive_title::text,
                estimated_hourly_rate_min,
                estimated_hourly_rate_max,
                hours_per_week,
                is_remote,
                posted_date,
                appeal_summary,
                url,
                slug
            FROM jobs
            WHERE is_active = true AND is_fractional = true
        """
        params = []
        param_idx = 1

        if executive_title:
            query += f" AND executive_title::text ILIKE ${param_idx}"
            params.append(f"%{executive_title}%")
            param_idx += 1

        query += f" ORDER BY posted_date DESC NULLS LAST LIMIT ${param_idx}"
        params.append(limit)

        results = await conn.fetch(query, *params)

        return [dict(r) for r in results]


async def get_job_by_slug(slug: str) -> Optional[dict]:
    """Get a specific job by its slug."""
    async with get_connection() as conn:
        result = await conn.fetchrow("""
            SELECT
                id::text,
                title,
                normalized_title,
                company_name,
                location,
                city::text,
                country,
                employment_type,
                workplace_type,
                is_remote,
                executive_title::text,
                role_category::text,
                industry::text,
                salary_min,
                salary_max,
                salary_currency,
                estimated_hourly_rate_min,
                estimated_hourly_rate_max,
                hours_per_week,
                description_snippet,
                full_description,
                appeal_summary,
                key_deliverables,
                skills_required,
                requirements,
                responsibilities,
                benefits,
                qualifications,
                about_company,
                posted_date,
                url,
                slug
            FROM jobs
            WHERE slug = $1 AND is_active = true
        """, slug)

        return dict(result) if result else None


async def store_user_query(
    user_id: str,
    query: str,
    job_id: Optional[str] = None,
    job_title: Optional[str] = None,
    session_id: Optional[str] = None
) -> None:
    """Store a user's query for history tracking."""
    if not user_id or not query:
        return

    # Note: This requires a user_queries table in the database
    # For now, just log it
    import sys
    print(f"[FQ Query] User {user_id}: {query[:50]}...", file=sys.stderr)
