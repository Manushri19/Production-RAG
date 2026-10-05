"""Tests for meridian.workers.chunker"""

import pytest
from meridian.workers.chunker import UnifiedChunk, UnifiedChunker


class TestUnifiedChunk:
    """Tests for the UnifiedChunk data class."""

    def test_creation(self):
        """Create a basic chunk."""
        chunk = UnifiedChunk(
            text="Hello world",
            chunk_type="text",
            page_no=1,
            order=0,
        )
        assert chunk.text == "Hello world"
        assert chunk.chunk_type == "text"
        assert chunk.page_no == 1
        assert chunk.order == 0
        assert chunk.headings == []
        assert chunk.metadata == {}

    def test_with_metadata(self):
        """Create a chunk with metadata."""
        chunk = UnifiedChunk(
            text="Table data",
            chunk_type="table",
            page_no=2,
            order=5,
            headings=["Section 1", "Subsection A"],
            metadata={"vlm_extracted": True},
        )
        assert chunk.headings == ["Section 1", "Subsection A"]
        assert chunk.metadata["vlm_extracted"] is True


class TestUnifiedChunker:
    """Tests for the UnifiedChunker class."""

    def setup_method(self):
        self.chunker = UnifiedChunker(max_text_chunk_chars=200, min_text_chunk_chars=50)

    def test_chunks_to_dict(self):
        """Convert chunks to dict format."""
        chunks = [
            UnifiedChunk(text="Hello", chunk_type="text", page_no=1, order=0),
            UnifiedChunk(text="World", chunk_type="text", page_no=1, order=1),
        ]
        result = self.chunker.chunks_to_dict(chunks)
        assert len(result) == 2
        assert result[0]["text"] == "Hello"
        assert result[0]["type"] == "text"
        assert result[0]["page_no"] == 1
        assert result[0]["order"] == 0

    def test_merge_small_chunks(self):
        """Small text chunks are merged with previous."""
        chunks = [
            UnifiedChunk(text="A" * 100, chunk_type="text", page_no=1, order=0),
            UnifiedChunk(text="B" * 30, chunk_type="text", page_no=1, order=1),  # Small
        ]
        result = self.chunker.merge_small_chunks(chunks)
        assert len(result) == 1
        assert "A" * 100 in result[0].text
        assert "B" * 30 in result[0].text

    def test_merge_small_chunks_preserves_non_text(self):
        """Non-text chunks are never merged."""
        chunks = [
            UnifiedChunk(text="Short", chunk_type="text", page_no=1, order=0),
            UnifiedChunk(text="[Table]", chunk_type="table", page_no=1, order=1),
            UnifiedChunk(text="Also short", chunk_type="text", page_no=1, order=2),
        ]
        result = self.chunker.merge_small_chunks(chunks)
        # Table stays separate, small text can't merge across it
        assert any(c.chunk_type == "table" for c in result)

    def test_merge_small_chunks_first_chunk_small(self):
        """First chunk being small doesn't cause issues."""
        chunks = [
            UnifiedChunk(text="X", chunk_type="text", page_no=1, order=0),
        ]
        result = self.chunker.merge_small_chunks(chunks)
        assert len(result) == 1

    def test_merge_text_chunks(self):
        """Consecutive text chunks are merged up to max."""
        chunks = [
            UnifiedChunk(text="A" * 80, chunk_type="text", page_no=1, order=0),
            UnifiedChunk(text="B" * 80, chunk_type="text", page_no=1, order=1),
        ]
        result = self.chunker.merge_text_chunks(chunks)
        assert len(result) == 1
        assert "A" * 80 in result[0].text
        assert "B" * 80 in result[0].text

    def test_merge_text_chunks_respects_max(self):
        """Text merging stops at max_chars."""
        chunks = [
            UnifiedChunk(text="A" * 150, chunk_type="text", page_no=1, order=0),
            UnifiedChunk(text="B" * 150, chunk_type="text", page_no=1, order=1),
        ]
        # max is 200, so these should NOT merge
        result = self.chunker.merge_text_chunks(chunks)
        assert len(result) == 2

    def test_merge_text_chunks_non_text_breaks(self):
        """Non-text chunks break text merging."""
        chunks = [
            UnifiedChunk(text="A" * 50, chunk_type="text", page_no=1, order=0),
            UnifiedChunk(text="[Table]", chunk_type="table", page_no=1, order=1),
            UnifiedChunk(text="B" * 50, chunk_type="text", page_no=1, order=2),
        ]
        result = self.chunker.merge_text_chunks(chunks)
        assert len(result) == 3

    def test_empty_chunks(self):
        """Empty chunk lists are handled."""
        assert self.chunker.merge_small_chunks([]) == []
        assert self.chunker.merge_text_chunks([]) == []
        assert self.chunker.chunks_to_dict([]) == []
