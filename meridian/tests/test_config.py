"""Tests for meridian.config"""

import os
import importlib
import pytest


def test_default_redis_url():
    """Test default Redis URL."""
    from meridian.config import REDIS_URL
    assert REDIS_URL == "redis://localhost:6379/0"


def test_default_content_block_thresholds():
    """Test default content block filtering thresholds."""
    from meridian.config import (
        CONTENT_BLOCK_MIN_WIDTH_PT,
        CONTENT_BLOCK_MIN_HEIGHT_PT,
        CONTENT_BLOCK_MIN_AREA_PT2,
        CONTENT_BLOCK_CONTAINMENT_THRESHOLD,
    )
    assert CONTENT_BLOCK_MIN_WIDTH_PT == 20
    assert CONTENT_BLOCK_MIN_HEIGHT_PT == 10
    assert CONTENT_BLOCK_MIN_AREA_PT2 == 500
    assert CONTENT_BLOCK_CONTAINMENT_THRESHOLD == 0.8


def test_default_image_scale():
    """Test default image scale."""
    from meridian.config import IMAGE_SCALE
    assert IMAGE_SCALE == 2.0


def test_default_table_padding():
    """Test default table padding."""
    from meridian.config import TABLE_PADDING_PX
    assert TABLE_PADDING_PX == 70


def test_default_service_port():
    """Test default service port."""
    from meridian.config import SERVICE_PORT
    assert SERVICE_PORT == 8001


def test_default_docling_instances():
    """Test default Docling instances."""
    from meridian.config import DOCLING_INSTANCES
    assert len(DOCLING_INSTANCES) == 8
    assert DOCLING_INSTANCES[0] == "http://localhost:8001"
    assert DOCLING_INSTANCES[-1] == "http://localhost:8008"


def test_default_vllm_model():
    """Test default VLM model."""
    from meridian.config import VLLM_MODEL
    assert VLLM_MODEL == "Qwen/Qwen3-VL-8B-Instruct"


def test_default_embedding_model():
    """Test default embedding model."""
    from meridian.config import EMBEDDING_MODEL
    assert EMBEDDING_MODEL == "qwen3-embedding:4b-q8_0"


def test_default_ocr_disabled():
    """Test OCR is disabled by default."""
    from meridian.config import OCR_ENABLED
    assert OCR_ENABLED is False


def test_default_formula_enrichment_disabled():
    """Test formula enrichment is disabled by default."""
    from meridian.config import FORMULA_ENRICHMENT_ENABLED
    assert FORMULA_ENRICHMENT_ENABLED is False


def test_default_table_structure_enabled():
    """Test table structure is enabled by default."""
    from meridian.config import TABLE_STRUCTURE_ENABLED
    assert TABLE_STRUCTURE_ENABLED is True


def test_ensure_temp_dir(tmp_path):
    """Test temp directory creation."""
    import meridian.config as config
    original = config.TEMP_DIR
    try:
        config.TEMP_DIR = tmp_path / "test_temp"
        result = config.ensure_temp_dir()
        assert result.exists()
    finally:
        config.TEMP_DIR = original
