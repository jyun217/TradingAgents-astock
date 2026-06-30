import os
import unittest
from unittest.mock import patch

import pytest

from tradingagents.llm_clients.endpoints import resolve_base_url


@pytest.mark.unit
class TestResolveBaseUrl(unittest.TestCase):
    def test_explicit_wins(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env/v1", "BACKEND_URL": "https://generic/v1"}, clear=False):
            self.assertEqual(resolve_base_url("openai", "https://explicit/v1"), "https://explicit/v1")

    def test_provider_env_over_backend(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://oai/v1", "BACKEND_URL": "https://generic/v1"}, clear=False):
            self.assertEqual(resolve_base_url("openai", None), "https://oai/v1")
        with patch.dict(os.environ, {"ANTHROPIC_BASE_URL": "https://ant", "BACKEND_URL": "https://generic/v1"}, clear=False):
            self.assertEqual(resolve_base_url("anthropic", None), "https://ant")

    def test_backend_url_fallback(self):
        with patch.dict(os.environ, {"BACKEND_URL": "https://generic/v1"}, clear=True):
            self.assertEqual(resolve_base_url("openai", None), "https://generic/v1")

    def test_none_when_nothing_set(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_base_url("openai", None))
            self.assertIsNone(resolve_base_url("openai", "   "))

    def test_unknown_provider_uses_backend_only(self):
        with patch.dict(os.environ, {"BACKEND_URL": "https://generic/v1"}, clear=True):
            self.assertEqual(resolve_base_url("deepseek", None), "https://generic/v1")
