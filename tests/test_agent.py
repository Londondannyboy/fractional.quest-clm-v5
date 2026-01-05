"""Tests for the Pydantic AI agent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# Import models - these don't require API key
from api.models import SearchResults, ArticleResult


@pytest.fixture
def mock_search_results():
    """Sample search results for testing."""
    return SearchResults(
        articles=[
            ArticleResult(
                id="1",
                title="Ignatius Sancho - The First Black Briton to Vote",
                content="""Ignatius Sancho was born on a slave ship in 1729.
                He went on to become the first Black person to vote in Britain.
                He ran a grocery shop in Westminster and was painted by Gainsborough.
                His letters were published and became bestsellers.""",
                score=0.92,
            ),
        ],
        query="ignatius sancho",
    )


@pytest.fixture
def mock_empty_search():
    """Empty search results."""
    return SearchResults(articles=[], query="unknown topic")


class TestGenerateResponse:
    """Test the response generation with validation."""

    @pytest.mark.asyncio
    async def test_generates_grounded_response(self, mock_search_results):
        """Agent should generate response grounded in search results."""
        from api.agent import generate_response

        # Import here to avoid loading agent at module level
        with patch("api.agent.get_vic_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_result = MagicMock()
            mock_result.data.response_text = (
                "Ah, Ignatius Sancho! He was born on a slave ship in 1729 "
                "and became the first Black person to vote in Britain."
            )

            # Make run() return an awaitable
            async def mock_run(*args, **kwargs):
                return mock_result

            mock_agent.run = mock_run
            mock_get_agent.return_value = mock_agent

            response = await generate_response("Tell me about Ignatius Sancho")

            assert "1729" in response
            assert "vote" in response.lower()

    @pytest.mark.asyncio
    async def test_handles_validation_failure_gracefully(self):
        """When validation fails, should return safe fallback."""
        from api.agent import generate_response
        from pydantic import ValidationError

        with patch("api.agent.get_vic_agent") as mock_get_agent:
            mock_agent = MagicMock()

            # Simulate validation failure (hallucination detected)
            async def mock_run(*args, **kwargs):
                raise ValidationError.from_exception_data(
                    "ValidatedVICResponse",
                    [{"type": "value_error", "msg": "architect not in source"}],
                )

            mock_agent.run = mock_run
            mock_get_agent.return_value = mock_agent

            response = await generate_response("Who designed the Royal Aquarium?")

            # Should return safe fallback, not raise
            # May match accurate/don't or trouble/different way
            assert any(phrase in response.lower() for phrase in [
                "accurate", "don't", "trouble", "different way"
            ])

    @pytest.mark.asyncio
    async def test_handles_unexpected_errors(self):
        """Unexpected errors should return friendly message."""
        from api.agent import generate_response

        with patch("api.agent.get_vic_agent") as mock_get_agent:
            mock_agent = MagicMock()

            async def mock_run(*args, **kwargs):
                raise Exception("Unexpected error")

            mock_agent.run = mock_run
            mock_get_agent.return_value = mock_agent

            response = await generate_response("Any question")

            assert "trouble" in response.lower() or "different way" in response.lower()


class TestPhoneticCorrections:
    """Test that phonetic corrections are applied."""

    def test_ignatius_corrections(self):
        from api.tools import normalize_query

        assert "ignatius" in normalize_query("Tell me about Ignacio Sancho")
        assert "ignatius" in normalize_query("Who was Ignacius?")

    def test_place_name_corrections(self):
        from api.tools import normalize_query

        assert "tyburn" in normalize_query("What happened at Tie Burn?")
        assert "thorney" in normalize_query("Tell me about Thorny Island")
        assert "thames" in normalize_query("the river tems")

    def test_preserves_correct_spelling(self):
        from api.tools import normalize_query

        result = normalize_query("Tell me about Ignatius Sancho")
        assert "ignatius" in result.lower()
