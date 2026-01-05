"""Tests for fact-grounding validation."""

import pytest
from pydantic import ValidationError

from api.models import ValidatedVICResponse, DeclinedResponse


class TestValidatedVICResponse:
    """Test the ValidatedVICResponse model validators."""

    def test_valid_response_with_facts_in_source(self):
        """Response with all facts grounded in source should pass."""
        response = ValidatedVICResponse(
            response_text="Ignatius Sancho was born in 1729 on a slave ship.",
            facts_stated=["born in 1729", "slave ship"],
            source_content="Ignatius Sancho was born on a slave ship in 1729 and became a notable figure.",
            source_titles=["Ignatius Sancho - The First Black Briton to Vote"],
        )
        assert response.response_text is not None
        assert len(response.facts_stated) == 2

    def test_rejects_fact_not_in_source(self):
        """Response with facts not in source should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ValidatedVICResponse(
                response_text="The building was designed by Christopher Wren.",
                facts_stated=["designed by Christopher Wren"],
                source_content="The Royal Aquarium opened in 1876 as an entertainment venue.",
                source_titles=["The Royal Aquarium"],
            )
        # Should fail either for fact grounding or architect attribution
        error_msg = str(exc_info.value).lower()
        assert "not grounded" in error_msg or "architect" in error_msg or "designed" in error_msg

    def test_rejects_architect_not_in_source(self):
        """Mentioning architect when not in source should fail."""
        with pytest.raises(ValidationError) as exc_info:
            ValidatedVICResponse(
                response_text="The building was designed by a famous architect.",
                facts_stated=[],
                source_content="The building opened in 1850 and was very popular.",
                source_titles=["Historic Building"],
            )
        assert "architect" in str(exc_info.value).lower() or "designed" in str(exc_info.value).lower()

    def test_rejects_hallucinated_year(self):
        """Years not in source should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ValidatedVICResponse(
                response_text="This happened in 1923.",
                facts_stated=["happened in 1923"],
                source_content="The event took place in the Victorian era.",
                source_titles=["Victorian Events"],
            )
        assert "1923" in str(exc_info.value)

    def test_allows_year_from_source(self):
        """Years present in source should pass."""
        response = ValidatedVICResponse(
            response_text="The building opened in 1876.",
            facts_stated=["opened in 1876"],
            source_content="The Royal Aquarium opened in 1876.",
            source_titles=["Royal Aquarium"],
        )
        assert "1876" in response.response_text

    def test_empty_response_rejected(self):
        """Empty response text should fail."""
        with pytest.raises(ValidationError):
            ValidatedVICResponse(
                response_text="",
                facts_stated=[],
                source_content="Some content",
                source_titles=["Title"],
            )

    def test_no_facts_with_no_source_is_ok(self):
        """Response with no facts and no source (declining) should pass."""
        response = ValidatedVICResponse(
            response_text="I don't have information about that in my articles.",
            facts_stated=[],
            source_content="",
            source_titles=[],
        )
        assert response.response_text is not None


class TestDeclinedResponse:
    """Test the DeclinedResponse model."""

    def test_valid_decline(self):
        """Properly worded decline should pass."""
        response = DeclinedResponse(
            response_text="I don't have that information in my articles.",
            reason="No matching articles found",
        )
        assert response.response_text is not None

    def test_decline_must_indicate_limitation(self):
        """Decline without indication phrase should fail."""
        with pytest.raises(ValidationError):
            DeclinedResponse(
                response_text="The building was constructed in 1850.",
                reason="No source",
            )


class TestArchitectHallucination:
    """Specific tests for architect/designer hallucination prevention."""

    @pytest.mark.parametrize("pattern", [
        "designed by",
        "built by",
        "architect",
        "constructed by",
        "created by",
    ])
    def test_rejects_attribution_not_in_source(self, pattern):
        """Various attribution patterns should fail if not in source."""
        with pytest.raises(ValidationError):
            ValidatedVICResponse(
                response_text=f"This was {pattern} someone important.",
                facts_stated=[],
                source_content="The building is located in Westminster.",
                source_titles=["Westminster Building"],
            )

    def test_allows_attribution_when_in_source(self):
        """Attribution should pass when source mentions it."""
        response = ValidatedVICResponse(
            response_text="The church was designed by Christopher Wren.",
            facts_stated=["designed by Christopher Wren"],
            source_content="St Paul's Cathedral was designed by Christopher Wren after the Great Fire.",
            source_titles=["St Paul's Cathedral"],
        )
        assert "Wren" in response.response_text
