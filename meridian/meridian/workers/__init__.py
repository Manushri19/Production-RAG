"""
Workers package for Meridian document processing.

CRITICAL: These workers must have ZERO Docling/pypdfium2 imports.
Only docling_core.types imports are allowed (for data types only).

Workers:
- box_annotator: Draw numbered boxes on formula pages (PIL only)
- chunker: Build chunks from DoclingDocument (data types only)
- document_worker: Main processing orchestration
- tasks: Celery task definitions
"""
