"""
Meridian - Large-scale GPU-accelerated document processing pipeline.

Architecture:
- GPU Services: Docling (PDF processing), vLLM (vision-language), Ollama (embeddings)
- CPU Workers: Stateless Celery workers calling services via HTTP
- Orchestrator: Batch submission, progress tracking, checkpoint/resume
- Vector Store: Qdrant for semantic search over processed documents
"""

__version__ = "0.1.0"
