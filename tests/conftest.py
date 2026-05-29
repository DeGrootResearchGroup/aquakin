"""Shared pytest fixtures."""

from pathlib import Path

import pytest

import aquakin

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_network():
    """Load the simple A -> B test network."""
    return aquakin.load_network_from_file(FIXTURES / "simple_network.yaml")
