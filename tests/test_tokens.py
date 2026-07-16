import pytest

from clip_token_lab.tokens import parse_token_ids


def test_parse_token_ids_accepts_mixed_separators():
    assert parse_token_ids("1, 2\n3 [4]") == [1, 2, 3, 4]


def test_parse_token_ids_checks_vocab():
    with pytest.raises(ValueError):
        parse_token_ids("1 99", vocab_size=10)
