import pytest
from analysis.stats import get_document_length


def test_get_document_length():
    assert get_document_length("Hello world") == 3