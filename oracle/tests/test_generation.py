"""
Tests for the generation pipeline.

Covers:
- Prompt builder: context block formatting, citation instructions
- Streamer: SSE event format, citation parsing, source enrichment
"""

import json
import pytest
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# Prompt Builder Tests
# ---------------------------------------------------------------------------

def test_system_prompt_contains_hard_constraints():
    """Verify that the system prompt has all required guardrails."""
    from oracle.generation.prompt_builder import SYSTEM_PROMPT

    required_phrases = [
        "CONTEXT-ONLY",
        "I cannot answer this question based on the provided documents",
        "inline citation",
        "SOURCES",
        "VERBATIM",
    ]
    for phrase in required_phrases:
        assert phrase in SYSTEM_PROMPT, f"System prompt missing required constraint: '{phrase}'"


def test_context_block_numbers_chunks():
    """Verify chunks are numbered starting from 1."""
    from oracle.generation.prompt_builder import build_context_block

    chunks = [
        {"text": "First chunk text.", "document_id": "doc_a", "page_no": 1, "chunk_type": "text", "headings": []},
        {"text": "Second chunk text.", "document_id": "doc_a", "page_no": 2, "chunk_type": "table", "headings": ["Results"]},
    ]

    block = build_context_block(chunks)
    assert "[1]" in block
    assert "[2]" in block
    assert "doc_a" in block
    assert "Page: 1" in block
    assert "Results" in block


def test_user_prompt_includes_question():
    """Verify user prompt includes both context and question."""
    from oracle.generation.prompt_builder import build_context_block, build_user_prompt

    chunks = [{"text": "Test chunk.", "document_id": "d1", "page_no": 1, "chunk_type": "text", "headings": []}]
    context = build_context_block(chunks)
    prompt = build_user_prompt(query="What is the melting point?", context_block=context)

    assert "What is the melting point?" in prompt
    assert "ANSWER" in prompt
    assert "[N]" in prompt  # Citation instruction


# ---------------------------------------------------------------------------
# Streamer Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streamer_emits_token_events():
    """Verify that token events are yielded for each LLM token."""
    from oracle.generation.streamer import stream_generation

    chunks = [
        {"text": "Ti-6Al-4V has a density of 4.43 g/cm³.", "document_id": "doc1", "page_no": 5,
         "chunk_type": "text", "headings": []},
    ]

    mock_llm = AsyncMock()

    async def mock_stream(*args, **kwargs):
        for token in ["The", " density", " is", " 4.43", " g/cm³", " [1]"]:
            yield token

    mock_llm.stream_completion = mock_stream

    events = []
    async for event in stream_generation(query="What is the density?", chunks=chunks, llm=mock_llm):
        events.append(event)

    token_events = [e for e in events if '"token"' in e]
    assert len(token_events) == 6

    # Verify SSE format
    for te in token_events:
        assert te.startswith("data: ")
        payload = json.loads(te[len("data: "):].strip())
        assert "token" in payload


@pytest.mark.asyncio
async def test_streamer_parses_sources_block():
    """Verify that the sources JSON block is parsed and emitted."""
    from oracle.generation.streamer import stream_generation

    chunks = [
        {"text": "Steel yield strength is 250 MPa.", "document_id": "steel_ds", "page_no": 3,
         "chunk_type": "text", "headings": []},
    ]

    mock_llm = AsyncMock()
    full_response = (
        'The yield strength is 250 MPa [1]. '
        '```sources\n'
        '[{"ref": 1, "document_id": "steel_ds", "page_no": 3, "chunk_type": "text", "excerpt": "Steel yield strength"}]\n'
        '```'
    )

    async def mock_stream(*args, **kwargs):
        for char in full_response:
            yield char

    mock_llm.stream_completion = mock_stream

    events = []
    async for event in stream_generation(query="Yield strength?", chunks=chunks, llm=mock_llm):
        events.append(event)

    source_events = [e for e in events if '"sources"' in e]
    assert len(source_events) == 1

    payload = json.loads(source_events[0][len("data: "):].strip())
    sources = payload["sources"]
    assert len(sources) == 1
    assert sources[0]["document_id"] == "steel_ds"
    assert sources[0]["page_no"] == 3


@pytest.mark.asyncio
async def test_streamer_emits_done_terminator():
    """Verify the stream always ends with [DONE]."""
    from oracle.generation.streamer import stream_generation

    chunks = [{"text": "text", "document_id": "d", "page_no": 1, "chunk_type": "text", "headings": []}]
    mock_llm = AsyncMock()

    async def mock_stream(*args, **kwargs):
        yield "Answer without sources block."

    mock_llm.stream_completion = mock_stream

    events = []
    async for event in stream_generation(query="q", chunks=chunks, llm=mock_llm):
        events.append(event)

    assert events[-1] == "data: [DONE]\n\n"
