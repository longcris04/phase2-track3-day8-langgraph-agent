"""Pytest fixtures / setup.

Load the project's .env so tests that exercise the real LLM (graph smoke tests)
can see OPENROUTER_API_KEY / OPENROUTER_MODEL without manual exporting.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()
