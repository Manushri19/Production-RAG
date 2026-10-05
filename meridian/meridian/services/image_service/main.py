"""
Image Service - Renders PDF regions as base64 images.

Two endpoints matching the agent tools exactly:
- POST /api/view_chunk_image - View picture/table chunks
- POST /api/get_visual_region - Visual verification of any region

Port: configured via IMAGE_SERVICE_PORT
"""

import base64
import logging
from pathlib import Path
from typing import Optional, List

import fitz  # PyMuPDF
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from meridian.config import PDF_DIR, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, IMAGE_SERVICE_PORT

# Configuration
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
DEFAULT_COLLECTION = QDRANT_COLLECTION
PORT = IMAGE_SERVICE_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Meridian Image Service",
    description="Renders PDF regions as base64 images",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant = QdrantClient(url=QDRANT_URL)


# ============================================================================
# Request/Response Models (matching agent tool schemas exactly)
# ============================================================================

class ViewChunkImageRequest(BaseModel):
    """Matches view_chunk_image tool input schema."""
    document_id: str
    chunk_order: Optional[int] = None
    chunk_orders: Optional[List[int]] = None  # Max 4
    collection: Optional[str] = None


class GetVisualRegionRequest(BaseModel):
    """Matches get_visual_region tool input schema."""
    document_id: str
    page_no: Optional[int] = None
    chunk_order: Optional[int] = None
    expand_above: int = 100
    expand_below: int = 100
    collection: Optional[str] = None


class ImageResult(BaseModel):
    """Single image result."""
    image_base64: str
    order: int
    page_no: int
    chunk_type: str
    figure_type: Optional[str] = None
    caption: Optional[str] = None
    is_complex_table: Optional[bool] = None
    text: Optional[str] = None


class ViewChunkImageResponse(BaseModel):
    """Response for view_chunk_image."""
    success: bool
    images: Optional[List[ImageResult]] = None
    errors: Optional[List[str]] = None
    error: Optional[str] = None


class GetVisualRegionResponse(BaseModel):
    """Response for get_visual_region."""
    success: bool
    image_base64: Optional[str] = None
    page_no: Optional[int] = None
    chunk_order: Optional[int] = None
    chunk_type: Optional[str] = None
    capture_type: Optional[str] = None  # "chunk_region" or "full_page"
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def find_pdf(document_id: str) -> Optional[Path]:
    """Find PDF file for a document ID."""
    pdf_path = PDF_DIR / f"{document_id}.pdf"
    if pdf_path.exists():
        return pdf_path

    for ext in [".pdf", ".PDF"]:
        alt_path = PDF_DIR / f"{document_id}{ext}"
        if alt_path.exists():
            return alt_path

    matches = list(PDF_DIR.glob(f"*{document_id}*.pdf"))
    if matches:
        return matches[0]

    return None


def render_chunk_region(
    doc: fitz.Document,
    page_no: int,
    bbox: dict,
    padding_percent: float = 0.05,
    scale: float = 2.0,
) -> bytes:
    """Render a chunk's bbox with padding."""
    page = doc[page_no - 1]
    page_rect = page.rect

    # Convert bbox (PDF bottom-left -> PyMuPDF top-left)
    left = bbox.get("l", 0)
    right = bbox.get("r", page_rect.width)
    top = bbox.get("t", page_rect.height)
    bottom = bbox.get("b", 0)

    x0 = left
    x1 = right
    y0 = page_rect.height - top
    y1 = page_rect.height - bottom

    # Add padding
    width = x1 - x0
    height = y1 - y0
    padding_x = width * padding_percent
    padding_y = height * padding_percent

    x0 = max(0, x0 - padding_x)
    y0 = max(0, y0 - padding_y)
    x1 = min(page_rect.width, x1 + padding_x)
    y1 = min(page_rect.height, y1 + padding_y)

    clip_rect = fitz.Rect(x0, y0, x1, y1)
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
    return pix.tobytes("png")


def render_expanded_region(
    doc: fitz.Document,
    page_no: int,
    bbox: dict,
    expand_above: int,
    expand_below: int,
    scale: float = 2.0,
) -> bytes:
    """Render a chunk's bbox with custom expansion."""
    page = doc[page_no - 1]
    page_rect = page.rect

    left = bbox.get("l", 0)
    right = bbox.get("r", page_rect.width)
    top = bbox.get("t", page_rect.height)
    bottom = bbox.get("b", 0)

    x0 = left
    x1 = right
    y0 = page_rect.height - top
    y1 = page_rect.height - bottom

    # Apply expansion
    y0 = max(0, y0 - expand_above)
    y1 = min(page_rect.height, y1 + expand_below)

    clip_rect = fitz.Rect(x0, y0, x1, y1)
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
    return pix.tobytes("png")


def render_full_page(doc: fitz.Document, page_no: int, scale: float = 2.0) -> bytes:
    """Render full page."""
    page = doc[page_no - 1]
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check."""
    pdf_count = len(list(PDF_DIR.glob("*.pdf"))) if PDF_DIR.exists() else 0
    return {
        "status": "healthy",
        "pdf_dir": str(PDF_DIR),
        "pdf_count": pdf_count,
        "qdrant_url": QDRANT_URL,
        "collection": DEFAULT_COLLECTION,
    }


@app.post("/api/view_chunk_image", response_model=ViewChunkImageResponse)
async def view_chunk_image(request: ViewChunkImageRequest):
    """
    View picture/table chunk images.

    Only works for chunk_type == "picture" or "table".
    Supports viewing up to 4 images at once.
    """
    try:
        collection = request.collection or DEFAULT_COLLECTION

        # Normalize to list of orders
        if request.chunk_orders is not None:
            orders_to_fetch = request.chunk_orders[:4]
        elif request.chunk_order is not None:
            orders_to_fetch = [request.chunk_order]
        else:
            return ViewChunkImageResponse(
                success=False,
                error="Either chunk_order or chunk_orders must be provided",
            )

        # Find PDF
        pdf_path = find_pdf(request.document_id)
        if not pdf_path:
            return ViewChunkImageResponse(
                success=False,
                error=f"PDF not found for document_id: {request.document_id}",
            )

        # Query Qdrant for all chunks
        results = qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchValue(value=request.document_id)),
                    FieldCondition(key="order", match=MatchAny(any=orders_to_fetch)),
                ]
            ),
            limit=len(orders_to_fetch),
            with_payload=True,
            with_vectors=False,
        )

        points = results[0]
        if not points:
            return ViewChunkImageResponse(
                success=False,
                error=f"No chunks found with document_id='{request.document_id}' and order(s)={orders_to_fetch}",
            )

        chunks_by_order = {p.payload.get("order"): p.payload for p in points}

        # Open PDF
        doc = fitz.open(str(pdf_path))
        images = []
        errors = []

        try:
            for order in orders_to_fetch:
                payload = chunks_by_order.get(order)
                if payload is None:
                    errors.append(f"Order {order}: Not found in document")
                    continue

                chunk_type = payload.get("chunk_type", "text")
                if chunk_type not in ("picture", "table"):
                    errors.append(f"Order {order}: Not a picture/table (type: {chunk_type})")
                    continue

                bbox = payload.get("meta_bbox")
                if not bbox:
                    errors.append(f"Order {order}: No bounding box stored")
                    continue

                page_no = payload.get("page_no", 1)
                if page_no < 1 or page_no > len(doc):
                    errors.append(f"Order {order}: Page {page_no} out of range")
                    continue

                # Render
                png_bytes = render_chunk_region(doc, page_no, bbox)
                image_base64 = base64.b64encode(png_bytes).decode("utf-8")

                images.append(ImageResult(
                    image_base64=image_base64,
                    order=order,
                    page_no=page_no,
                    chunk_type=chunk_type,
                    figure_type=payload.get("figure_type") or payload.get("meta_figure_type"),
                    caption=payload.get("meta_caption"),
                    is_complex_table=payload.get("is_complex_table") or payload.get("meta_complex"),
                    text=payload.get("text", "")[:500],
                ))

        finally:
            doc.close()

        if not images:
            return ViewChunkImageResponse(
                success=False,
                error="Could not process any images",
                errors=errors,
            )

        return ViewChunkImageResponse(
            success=True,
            images=images,
            errors=errors if errors else None,
        )

    except Exception as e:
        logger.exception(f"view_chunk_image failed: {e}")
        return ViewChunkImageResponse(
            success=False,
            error=str(e),
        )


@app.post("/api/get_visual_region", response_model=GetVisualRegionResponse)
async def get_visual_region(request: GetVisualRegionRequest):
    """
    Capture visual region for verification.

    Two modes:
    - With chunk_order: Centers on chunk's bbox with expansion
    - Without chunk_order: Full page (requires page_no)

    Works with ANY chunk type (text, picture, table, formula).
    """
    try:
        collection = request.collection or DEFAULT_COLLECTION

        if request.page_no is None and request.chunk_order is None:
            return GetVisualRegionResponse(
                success=False,
                error="Either page_no or chunk_order must be provided",
            )

        # Find PDF
        pdf_path = find_pdf(request.document_id)
        if not pdf_path:
            return GetVisualRegionResponse(
                success=False,
                error=f"PDF not found for document_id: {request.document_id}",
            )

        # If chunk_order provided, look up chunk
        chunk_payload = None
        page_no = request.page_no

        if request.chunk_order is not None:
            results = qdrant.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="document_id", match=MatchValue(value=request.document_id)),
                        FieldCondition(key="order", match=MatchValue(value=request.chunk_order)),
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            points = results[0]
            if not points:
                return GetVisualRegionResponse(
                    success=False,
                    error=f"Chunk not found: document_id={request.document_id}, order={request.chunk_order}",
                )

            chunk_payload = points[0].payload
            if page_no is None:
                page_no = chunk_payload.get("page_no")
                if page_no is None:
                    return GetVisualRegionResponse(
                        success=False,
                        error=f"Chunk at order {request.chunk_order} has no page_no stored",
                    )

        # Open PDF and render
        doc = fitz.open(str(pdf_path))

        try:
            if page_no < 1 or page_no > len(doc):
                return GetVisualRegionResponse(
                    success=False,
                    error=f"Page {page_no} out of range (document has {len(doc)} pages)",
                )

            if request.chunk_order is not None and chunk_payload is not None:
                bbox = chunk_payload.get("meta_bbox")
                if not bbox:
                    return GetVisualRegionResponse(
                        success=False,
                        error=f"Chunk at order {request.chunk_order} has no bounding box stored",
                    )

                png_bytes = render_expanded_region(
                    doc, page_no, bbox,
                    request.expand_above, request.expand_below
                )
                capture_type = "chunk_region"
                chunk_type = chunk_payload.get("chunk_type")
            else:
                png_bytes = render_full_page(doc, page_no)
                capture_type = "full_page"
                chunk_type = None

            image_base64 = base64.b64encode(png_bytes).decode("utf-8")

            return GetVisualRegionResponse(
                success=True,
                image_base64=image_base64,
                page_no=page_no,
                chunk_order=request.chunk_order,
                chunk_type=chunk_type,
                capture_type=capture_type,
            )

        finally:
            doc.close()

    except Exception as e:
        logger.exception(f"get_visual_region failed: {e}")
        return GetVisualRegionResponse(
            success=False,
            error=str(e),
        )


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
