"""Tests for header analyzer."""

from mdio.ingestion.header_analyzer import HeaderAnalyzer
from mdio.ingestion.header_analyzer import HeaderRequirements


def test_header_requirements():
    req = HeaderRequirements(required_fields={"inline", "crossline"})
    assert req.all_fields() == {"inline", "crossline"}
