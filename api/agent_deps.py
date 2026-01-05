"""
Fractional Quest Agent Dependencies - Runtime context for Pydantic AI agents.

This dataclass is passed to all agent tools and allows them to access
session context, user information, and conversation history.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FQAgentDeps:
    """Runtime dependencies for the Fractional Quest agent.

    Attributes:
        user_id: Unique identifier for the user (from Neon Auth)
        session_id: Current conversation session ID
        user_name: User's first name for personalization
        conversation_history: Recent messages for context
        enrichment_mode: True when running background enrichment (slower path)
        prior_roles: Role types discussed in previous turns (CFO, CMO, etc.)
        prior_companies: Companies mentioned in previous turns
        preferred_location: User's preferred work location (if known)
        preferred_remote: User's remote work preference (if known)
    """
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    user_name: Optional[str] = None
    conversation_history: list[dict] = field(default_factory=list)
    enrichment_mode: bool = False
    prior_roles: list[str] = field(default_factory=list)
    prior_companies: list[str] = field(default_factory=list)
    preferred_location: Optional[str] = None
    preferred_remote: Optional[bool] = None

    def has_context(self) -> bool:
        """Check if we have prior context from previous turns."""
        return bool(self.prior_roles or self.prior_companies)

    def add_role(self, role: str) -> None:
        """Add a role type to the context (deduped)."""
        if role and role not in self.prior_roles:
            self.prior_roles.append(role)

    def add_company(self, company: str) -> None:
        """Add a company to the context (deduped)."""
        if company and company not in self.prior_companies:
            self.prior_companies.append(company)

    def set_location_preference(self, location: str) -> None:
        """Set the user's preferred work location."""
        self.preferred_location = location

    def set_remote_preference(self, is_remote: bool) -> None:
        """Set the user's remote work preference."""
        self.preferred_remote = is_remote
