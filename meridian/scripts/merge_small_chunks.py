#!/usr/bin/env python3
"""
Merge text chunks in Qdrant collection to reach target size (512 chars).

This script:
1. Iterates through all documents in the collection
2. For each document, finds consecutive text chunks with same headings
3. Keeps merging text chunks until target size (512) is reached
4. Stops merging if max size (2000) would be exceeded or section boundary hit
5. Re-embeds merged chunks using Ollama
6. Updates Qdrant (deletes old, inserts merged)

Tables, pictures, and formulas are NEVER touched.
Section boundaries (different headings) are respected.
If target can't be reached (e.g., tables in between) that's fine.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk from Qdrant."""
    id: str
    document_id: str
    text: str
    chunk_type: str
    page_no: int
    order: int
    headings: list
    payload: dict  # Full payload for non-text chunks

    @property
    def text_length(self) -> int:
        return len(self.text)


@dataclass
class MergeStats:
    """Statistics for the merge operation."""
    documents_processed: int = 0
    text_chunks_before: int = 0
    text_chunks_after: int = 0
    chunks_merged: int = 0
    chunks_deleted: int = 0
    chunks_created: int = 0
    non_text_chunks: int = 0
    errors: int = 0


class ChunkMerger:
    """Merges small text chunks in a Qdrant collection."""

    def __init__(
        self,
        collection_name: str,
        qdrant_url: str = "http://localhost:6333",
        ollama_url: str = "http://localhost:11434",
        embed_model: str = "qwen3-embedding:4b-q8_0",
        target_chars: int = 512,
        max_merged_chars: int = 2000,
        dry_run: bool = False,
        batch_size: int = 100,
    ):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.target_chars = target_chars  # Try to reach this size
        self.max_merged_chars = max_merged_chars
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.stats = MergeStats()

        self.qdrant_client = httpx.Client(base_url=qdrant_url, timeout=60.0)
        self.ollama_client = httpx.Client(base_url=ollama_url, timeout=120.0)

    def get_all_document_ids(self) -> list[str]:
        """Get all unique document IDs in the collection."""
        logger.info("Fetching all document IDs...")

        document_ids = set()
        offset = None

        while True:
            payload = {
                "limit": 1000,
                "with_payload": {"include": ["document_id"]},
                "with_vector": False,
            }
            if offset:
                payload["offset"] = offset

            response = self.qdrant_client.post(
                f"/collections/{self.collection_name}/points/scroll",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            points = data["result"]["points"]
            if not points:
                break

            for point in points:
                doc_id = point["payload"].get("document_id")
                if doc_id:
                    document_ids.add(doc_id)

            offset = data["result"].get("next_page_offset")
            if not offset:
                break

            if len(document_ids) % 1000 == 0:
                logger.info(f"  Found {len(document_ids)} documents so far...")

        logger.info(f"Found {len(document_ids)} unique documents")
        return sorted(document_ids)

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        """Get all chunks for a document."""
        chunks = []
        offset = None

        while True:
            payload = {
                "limit": 500,
                "with_payload": True,
                "with_vector": False,
                "filter": {
                    "must": [
                        {"key": "document_id", "match": {"value": document_id}}
                    ]
                }
            }
            if offset:
                payload["offset"] = offset

            response = self.qdrant_client.post(
                f"/collections/{self.collection_name}/points/scroll",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            points = data["result"]["points"]
            if not points:
                break

            for point in points:
                p = point["payload"]
                chunks.append(Chunk(
                    id=point["id"],
                    document_id=p.get("document_id", ""),
                    text=p.get("text", ""),
                    chunk_type=p.get("chunk_type", "text"),
                    page_no=p.get("page_no", 0),
                    order=p.get("order", 0),
                    headings=p.get("headings", []),
                    payload=p,
                ))

            offset = data["result"].get("next_page_offset")
            if not offset:
                break

        return chunks

    def group_consecutive_text_chunks(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """
        Group consecutive text chunks with same headings.
        Non-text chunks break the sequence.
        """
        # Sort by order
        sorted_chunks = sorted(chunks, key=lambda c: c.order)

        groups = []
        current_group = []
        current_headings = None
        last_order = None

        for chunk in sorted_chunks:
            # Non-text chunks break the sequence
            if chunk.chunk_type != "text":
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_headings = None
                    last_order = None
                continue

            # Check if this continues the current group
            is_consecutive = (last_order is None or chunk.order == last_order + 1)
            same_headings = (current_headings is None or chunk.headings == current_headings)

            # Handle empty headings - treat two empty headings as "same"
            if current_headings == [] and chunk.headings == []:
                same_headings = True

            if is_consecutive and same_headings:
                current_group.append(chunk)
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [chunk]

            current_headings = chunk.headings
            last_order = chunk.order

        # Don't forget the last group
        if current_group:
            groups.append(current_group)

        return groups

    def merge_group(self, group: list[Chunk]) -> list[Chunk]:
        """
        Merge chunks in a group to try to reach target_chars (512).
        Keep merging until we reach the target or hit max_merged_chars.
        Returns list of resulting chunks (may be fewer than input).
        """
        if len(group) <= 1:
            return group

        result = []
        current_merged = None
        merged_ids = []
        merged_orders = []

        for chunk in group:
            if current_merged is None:
                # Start new merge group
                current_merged = chunk
                merged_ids = [chunk.id]
                merged_orders = [chunk.order]
            else:
                combined_length = current_merged.text_length + chunk.text_length + 2  # +2 for \n\n

                # Keep merging if:
                # 1. Current chunk is below target AND
                # 2. Combined would not exceed max
                if current_merged.text_length < self.target_chars and combined_length <= self.max_merged_chars:
                    # Merge them
                    new_text = current_merged.text + "\n\n" + chunk.text
                    merged_ids.append(chunk.id)
                    merged_orders.append(chunk.order)

                    current_merged = Chunk(
                        id=str(uuid.uuid4()),  # New ID for merged chunk
                        document_id=current_merged.document_id,
                        text=new_text,
                        chunk_type="text",
                        page_no=current_merged.page_no,  # Keep first chunk's page
                        order=current_merged.order,  # Keep first chunk's order
                        headings=current_merged.headings,
                        payload={
                            **current_merged.payload,
                            "text": new_text,
                            "merged": True,
                            "merged_from": list(merged_ids),
                            "original_orders": list(merged_orders),
                        }
                    )
                else:
                    # Current is at/above target OR combined would be too big
                    # Emit current and start new group
                    result.append(current_merged)
                    current_merged = chunk
                    merged_ids = [chunk.id]
                    merged_orders = [chunk.order]

        # Don't forget the last one
        if current_merged:
            result.append(current_merged)

        return result

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Ollama."""
        if not texts:
            return []

        response = self.ollama_client.post(
            "/api/embed",
            json={
                "model": self.embed_model,
                "input": texts,
                "keep_alive": "240h",
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]

    def delete_points(self, point_ids: list[str]):
        """Delete points from Qdrant."""
        if not point_ids:
            return

        response = self.qdrant_client.post(
            f"/collections/{self.collection_name}/points/delete",
            json={"points": point_ids}
        )
        response.raise_for_status()

    def upsert_points(self, chunks: list[Chunk], embeddings: list[list[float]]):
        """Upsert points to Qdrant."""
        if not chunks:
            return

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Build payload
            payload = {
                "document_id": chunk.document_id,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type,
                "page_no": chunk.page_no,
                "order": chunk.order,
                "headings": chunk.headings,
            }

            # Copy over other metadata from original payload
            for key, value in chunk.payload.items():
                if key not in payload and not key.startswith("_"):
                    payload[key] = value

            points.append({
                "id": chunk.id,
                "vector": embedding,
                "payload": payload,
            })

        response = self.qdrant_client.put(
            f"/collections/{self.collection_name}/points",
            json={"points": points}
        )
        response.raise_for_status()

    def process_document(self, document_id: str) -> tuple[int, int, int]:
        """
        Process a single document.
        Returns: (chunks_deleted, chunks_created, text_chunks_before)
        """
        chunks = self.get_document_chunks(document_id)

        # Separate text and non-text
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        non_text_chunks = [c for c in chunks if c.chunk_type != "text"]

        self.stats.non_text_chunks += len(non_text_chunks)
        text_before = len(text_chunks)

        if not text_chunks:
            return 0, 0, 0

        # Group consecutive text chunks with same headings
        groups = self.group_consecutive_text_chunks(chunks)

        # Merge small chunks in each group
        merged_groups = [self.merge_group(group) for group in groups]
        merged_chunks = [c for group in merged_groups for c in group]

        text_after = len(merged_chunks)

        # Find which chunks changed
        original_ids = {c.id for c in text_chunks}
        new_ids = {c.id for c in merged_chunks}

        # IDs to delete (original chunks that were merged)
        ids_to_delete = [c.id for c in text_chunks if c.id not in new_ids]

        # Chunks to create (new merged chunks)
        chunks_to_create = [c for c in merged_chunks if c.id not in original_ids]

        if not ids_to_delete and not chunks_to_create:
            # No changes needed
            return 0, 0, text_before

        if self.dry_run:
            return len(ids_to_delete), len(chunks_to_create), text_before

        # Embed new chunks
        if chunks_to_create:
            texts = [c.text for c in chunks_to_create]
            embeddings = self.embed_texts(texts)
            self.upsert_points(chunks_to_create, embeddings)

        # Delete old chunks
        if ids_to_delete:
            self.delete_points(ids_to_delete)

        return len(ids_to_delete), len(chunks_to_create), text_before

    def run(self):
        """Run the merge operation."""
        start_time = time.time()

        logger.info(f"Starting chunk merge on collection: {self.collection_name}")
        logger.info(f"Settings: target_chars={self.target_chars}, max_merged_chars={self.max_merged_chars}")
        logger.info(f"Dry run: {self.dry_run}")

        document_ids = self.get_all_document_ids()
        total_docs = len(document_ids)

        for i, doc_id in enumerate(document_ids):
            try:
                deleted, created, text_before = self.process_document(doc_id)

                self.stats.documents_processed += 1
                self.stats.chunks_deleted += deleted
                self.stats.chunks_created += created
                self.stats.text_chunks_before += text_before
                self.stats.text_chunks_after += (text_before - deleted + created)

                if deleted > 0 or created > 0:
                    self.stats.chunks_merged += deleted

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (total_docs - i - 1) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {i+1}/{total_docs} docs ({100*(i+1)/total_docs:.1f}%) | "
                        f"Deleted: {self.stats.chunks_deleted} | Created: {self.stats.chunks_created} | "
                        f"ETA: {eta/60:.1f}m"
                    )

            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
                self.stats.errors += 1

        elapsed = time.time() - start_time

        # Print summary
        logger.info("=" * 60)
        logger.info("MERGE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Documents processed: {self.stats.documents_processed}")
        logger.info(f"Text chunks before: {self.stats.text_chunks_before}")
        logger.info(f"Text chunks after: {self.stats.text_chunks_after}")
        logger.info(f"Chunks deleted: {self.stats.chunks_deleted}")
        logger.info(f"Chunks created: {self.stats.chunks_created}")
        logger.info(f"Net reduction: {self.stats.chunks_deleted - self.stats.chunks_created}")
        logger.info(f"Non-text chunks (untouched): {self.stats.non_text_chunks}")
        logger.info(f"Errors: {self.stats.errors}")
        logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")

        if self.dry_run:
            logger.info("\n*** DRY RUN - No changes were made ***")


def main():
    parser = argparse.ArgumentParser(description="Merge text chunks in Qdrant to reach target size")
    parser.add_argument("collection", help="Qdrant collection name")
    parser.add_argument("--target-chars", type=int, default=512,
                        help="Target characters per chunk - keep merging until this (default: 512)")
    parser.add_argument("--max-merged-chars", type=int, default=2000,
                        help="Maximum characters for merged chunk (default: 2000)")
    parser.add_argument("--embed-model", default="qwen3-embedding:4b-q8_0",
                        help="Ollama embedding model (default: qwen3-embedding:4b-q8_0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without modifying data")
    parser.add_argument("--qdrant-url", default="http://localhost:6333",
                        help="Qdrant URL")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama URL")

    args = parser.parse_args()

    merger = ChunkMerger(
        collection_name=args.collection,
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        embed_model=args.embed_model,
        target_chars=args.target_chars,
        max_merged_chars=args.max_merged_chars,
        dry_run=args.dry_run,
    )

    merger.run()


if __name__ == "__main__":
    main()
