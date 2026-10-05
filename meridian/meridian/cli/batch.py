"""
Batch Commands - Power user batch processing via Celery.

Wraps the existing orchestrator for backwards compatibility.
"""

import sys
from pathlib import Path

from meridian.orchestrator.batch_orchestrator import BatchOrchestrator
from meridian.orchestrator.progress import print_progress, print_batch_list, print_failed_docs


def handle_batch_command(args):
    """Route batch subcommands."""
    if not args.batch_command:
        print("Usage: meridian batch <command>")
        print("Commands: submit, status, resume, retry, pause, failed, clear, list")
        sys.exit(1)

    handler = {
        "submit": _cmd_submit,
        "status": _cmd_status,
        "resume": _cmd_resume,
        "retry": _cmd_retry,
        "pause": _cmd_pause,
        "failed": _cmd_failed,
        "clear": _cmd_clear,
        "list": _cmd_list,
    }.get(args.batch_command)

    if handler:
        handler(args)
    else:
        print(f"Unknown batch command: {args.batch_command}")
        sys.exit(1)


def _cmd_submit(args):
    orch = BatchOrchestrator()
    try:
        batch_id = orch.submit_batch(
            input_dir=Path(args.input_dir),
            collection=args.collection,
            batch_id=getattr(args, "batch_id", None),
            extract_tables=not args.no_tables,
            extract_figures=not args.no_figures,
            extract_formulas=not args.no_formulas,
            store_embeddings=not args.no_embeddings,
        )
        print(f"\nBatch submitted: {batch_id}")
        print(f"Collection: {args.collection}")
        print(f"\nTrack with: meridian batch status {batch_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_status(args):
    orch = BatchOrchestrator()
    if args.batch_id:
        progress = orch.get_progress(args.batch_id)
        if "error" in progress:
            print(f"Error: {progress['error']}", file=sys.stderr)
            sys.exit(1)
        print_progress(progress)
    else:
        batches = orch.list_batches()
        if not batches:
            print("No batches found.")
        else:
            print_batch_list(batches)


def _cmd_resume(args):
    orch = BatchOrchestrator()
    try:
        stale_timeout = 0 if args.force else 600
        count = orch.resume(args.batch_id, stale_timeout=stale_timeout)
        print(f"Resumed {count} tasks for batch {args.batch_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_retry(args):
    orch = BatchOrchestrator()
    try:
        count = orch.retry_failed(args.batch_id)
        print(f"Retrying {count} failed documents for batch {args.batch_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_pause(args):
    orch = BatchOrchestrator()
    try:
        orch.pause(args.batch_id)
        print(f"Paused batch {args.batch_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_failed(args):
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


def _cmd_clear(args):
    orch = BatchOrchestrator()
    if not args.force:
        confirm = input(f"Clear all state for batch '{args.batch_id}'? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return
    try:
        orch.clear_batch(args.batch_id)
        print(f"Cleared batch {args.batch_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_list(args):
    orch = BatchOrchestrator()
    batches = orch.list_batches()
    if not batches:
        print("No batches found.")
    else:
        print_batch_list(batches)
