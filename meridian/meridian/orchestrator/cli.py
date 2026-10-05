#!/usr/bin/env python3
"""
Orchestrator CLI

Command-line interface for batch document processing.

Usage:
    meridian submit <input_dir> [--collection NAME] [--batch-id ID]
    meridian status [BATCH_ID]
    meridian resume <batch_id>
    meridian retry <batch_id>
    meridian pause <batch_id>
    meridian failed <batch_id>
    meridian clear <batch_id>
    meridian list
"""

import argparse
import sys
from pathlib import Path

from meridian.orchestrator.batch_orchestrator import BatchOrchestrator
from meridian.orchestrator.progress import print_progress, print_batch_list, print_failed_docs


def cmd_submit(args):
    """Submit a batch of documents."""
    orch = BatchOrchestrator()

    try:
        batch_id = orch.submit_batch(
            input_dir=Path(args.input_dir),
            collection=args.collection,
            batch_id=args.batch_id,
            extract_tables=not args.no_tables,
            extract_figures=not args.no_figures,
            extract_formulas=not args.no_formulas,
            store_embeddings=not args.no_embeddings,
        )
        print(f"\nBatch submitted: {batch_id}")
        print(f"Collection: {args.collection}")
        print(f"\nUse 'meridian status {batch_id}' to track progress")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    """Show batch status."""
    orch = BatchOrchestrator()

    if args.batch_id:
        # Show specific batch
        progress = orch.get_progress(args.batch_id)
        if "error" in progress:
            print(f"Error: {progress['error']}", file=sys.stderr)
            sys.exit(1)
        print_progress(progress)
    else:
        # Show all batches
        batches = orch.list_batches()
        if not batches:
            print("No batches found.")
        else:
            print_batch_list(batches)


def cmd_resume(args):
    """Resume a paused or interrupted batch."""
    orch = BatchOrchestrator()

    try:
        # If --force, use 0 timeout to recover any processing docs immediately
        stale_timeout = 0 if args.force else 600
        count = orch.resume(args.batch_id, stale_timeout=stale_timeout)
        print(f"Resumed {count} tasks for batch {args.batch_id}")
        if args.force:
            print("(forced recovery of processing docs)")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_retry(args):
    """Retry failed documents."""
    orch = BatchOrchestrator()

    try:
        count = orch.retry_failed(args.batch_id)
        print(f"Retrying {count} failed documents for batch {args.batch_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_pause(args):
    """Pause batch processing."""
    orch = BatchOrchestrator()

    try:
        orch.pause(args.batch_id)
        print(f"Paused batch {args.batch_id}")
        print("Note: Running tasks will complete. Use 'resume' to continue.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_failed(args):
    """Show failed documents."""
    orch = BatchOrchestrator()

    try:
        failed = orch.get_failed_docs(args.batch_id)
        if not failed:
            print("No failed documents.")
        else:
            print_failed_docs(failed, limit=args.limit)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_clear(args):
    """Clear batch state."""
    orch = BatchOrchestrator()

    if not args.force:
        confirm = input(f"Clear all state for batch '{args.batch_id}'? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    try:
        orch.clear_batch(args.batch_id)
        print(f"Cleared batch {args.batch_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list(args):
    """List all batches."""
    orch = BatchOrchestrator()
    batches = orch.list_batches()

    if not batches:
        print("No batches found.")
    else:
        print_batch_list(batches)


def main():
    parser = argparse.ArgumentParser(
        description="Meridian CLI for batch document processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # submit
    p_submit = subparsers.add_parser("submit", help="Submit a batch of documents")
    p_submit.add_argument("input_dir", help="Directory containing PDF files")
    p_submit.add_argument("--collection", "-c", default="meridian_documents",
                          help="Qdrant collection name")
    p_submit.add_argument("--batch-id", "-b", help="Custom batch ID")
    p_submit.add_argument("--no-tables", action="store_true",
                          help="Skip table extraction")
    p_submit.add_argument("--no-figures", action="store_true",
                          help="Skip figure extraction")
    p_submit.add_argument("--no-formulas", action="store_true",
                          help="Skip formula extraction")
    p_submit.add_argument("--no-embeddings", action="store_true",
                          help="Skip embedding generation")
    p_submit.set_defaults(func=cmd_submit)

    # status
    p_status = subparsers.add_parser("status", help="Show batch status")
    p_status.add_argument("batch_id", nargs="?", help="Batch ID (shows all if omitted)")
    p_status.set_defaults(func=cmd_status)

    # resume
    p_resume = subparsers.add_parser("resume", help="Resume a batch")
    p_resume.add_argument("batch_id", help="Batch ID to resume")
    p_resume.add_argument("--force", "-f", action="store_true",
                          help="Force recover docs stuck in processing state")
    p_resume.set_defaults(func=cmd_resume)

    # retry
    p_retry = subparsers.add_parser("retry", help="Retry failed documents")
    p_retry.add_argument("batch_id", help="Batch ID")
    p_retry.set_defaults(func=cmd_retry)

    # pause
    p_pause = subparsers.add_parser("pause", help="Pause a batch")
    p_pause.add_argument("batch_id", help="Batch ID to pause")
    p_pause.set_defaults(func=cmd_pause)

    # failed
    p_failed = subparsers.add_parser("failed", help="Show failed documents")
    p_failed.add_argument("batch_id", help="Batch ID")
    p_failed.add_argument("--limit", "-n", type=int, default=20,
                          help="Max documents to show")
    p_failed.set_defaults(func=cmd_failed)

    # clear
    p_clear = subparsers.add_parser("clear", help="Clear batch state")
    p_clear.add_argument("batch_id", help="Batch ID to clear")
    p_clear.add_argument("--force", "-f", action="store_true",
                         help="Skip confirmation")
    p_clear.set_defaults(func=cmd_clear)

    # list
    p_list = subparsers.add_parser("list", help="List all batches")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
