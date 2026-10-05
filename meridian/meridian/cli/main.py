#!/usr/bin/env python3
"""
Meridian CLI - GPU-accelerated document processing.

Usage:
    meridian init                    # Auto-detect GPU, generate config
    meridian start                   # Start all services
    meridian stop                    # Stop all services
    meridian status                  # Health check all services
    meridian parse <path> [options]  # Process documents
    meridian serve [--port N]        # Start HTTP API

    # Batch orchestration (power users)
    meridian batch submit <dir>
    meridian batch status [id]
    meridian batch resume/retry/pause/list
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="meridian",
        description="Meridian - GPU-accelerated document processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  meridian init                          Auto-detect GPU, generate .env config
  meridian start                         Start all services
  meridian parse document.pdf            Process a single PDF
  meridian parse /path/to/pdfs/ -o out/  Process directory, output to out/
  meridian serve                         Start HTTP API on port 8080
  meridian batch submit /path/to/pdfs/   Submit large batch via Celery
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # --- init ---
    p_init = subparsers.add_parser("init", help="Auto-detect GPU and generate configuration")
    p_init.add_argument("--force", "-f", action="store_true", help="Overwrite existing .env")

    # --- start ---
    p_start = subparsers.add_parser("start", help="Start all services")
    p_start.add_argument("--service", "-s", choices=["docling", "vllm", "worker"], help="Start only a specific service")

    # --- stop ---
    p_stop = subparsers.add_parser("stop", help="Stop all services")

    # --- status ---
    p_status = subparsers.add_parser("status", help="Check service health")

    # --- parse ---
    p_parse = subparsers.add_parser("parse", help="Process PDF documents")
    p_parse.add_argument("path", help="PDF file or directory of PDFs")
    p_parse.add_argument("--output", "-o", default="./output", help="Output directory (default: ./output)")
    p_parse.add_argument("--no-tables", action="store_true", help="Skip table extraction")
    p_parse.add_argument("--no-figures", action="store_true", help="Skip figure extraction")
    p_parse.add_argument("--no-formulas", action="store_true", help="Skip formula extraction")
    p_parse.add_argument("--store", action="store_true", help="Also store in Qdrant vector database")
    p_parse.add_argument("--collection", "-c", default="meridian_documents", help="Qdrant collection name (with --store)")
    p_parse.add_argument("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: auto)")

    # --- serve ---
    p_serve = subparsers.add_parser("serve", help="Start HTTP API server")
    p_serve.add_argument("--port", "-p", type=int, default=8080, help="API port (default: 8080)")
    p_serve.add_argument("--host", default="0.0.0.0", help="API host (default: 0.0.0.0)")

    # --- batch (subcommand group) ---
    p_batch = subparsers.add_parser("batch", help="Batch processing commands (power users)")
    batch_sub = p_batch.add_subparsers(dest="batch_command", help="Batch command")

    bp_submit = batch_sub.add_parser("submit", help="Submit a batch of documents")
    bp_submit.add_argument("input_dir", help="Directory containing PDF files")
    bp_submit.add_argument("--collection", "-c", default="meridian_documents", help="Qdrant collection name")
    bp_submit.add_argument("--batch-id", "-b", help="Custom batch ID")
    bp_submit.add_argument("--no-tables", action="store_true", help="Skip table extraction")
    bp_submit.add_argument("--no-figures", action="store_true", help="Skip figure extraction")
    bp_submit.add_argument("--no-formulas", action="store_true", help="Skip formula extraction")
    bp_submit.add_argument("--no-embeddings", action="store_true", help="Skip embedding generation")

    bp_status = batch_sub.add_parser("status", help="Show batch status")
    bp_status.add_argument("batch_id", nargs="?", help="Batch ID (shows all if omitted)")

    bp_resume = batch_sub.add_parser("resume", help="Resume a batch")
    bp_resume.add_argument("batch_id", help="Batch ID to resume")
    bp_resume.add_argument("--force", "-f", action="store_true", help="Force recover stuck docs")

    bp_retry = batch_sub.add_parser("retry", help="Retry failed documents")
    bp_retry.add_argument("batch_id", help="Batch ID")

    bp_pause = batch_sub.add_parser("pause", help="Pause a batch")
    bp_pause.add_argument("batch_id", help="Batch ID to pause")

    bp_failed = batch_sub.add_parser("failed", help="Show failed documents")
    bp_failed.add_argument("batch_id", help="Batch ID")
    bp_failed.add_argument("--limit", "-n", type=int, default=20, help="Max documents to show")

    bp_clear = batch_sub.add_parser("clear", help="Clear batch state")
    bp_clear.add_argument("batch_id", help="Batch ID to clear")
    bp_clear.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    batch_sub.add_parser("list", help="List all batches")

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handlers
    if args.command == "init":
        from meridian.cli.init import cmd_init
        cmd_init(args)

    elif args.command == "start":
        from meridian.cli.services import cmd_start
        cmd_start(args)

    elif args.command == "stop":
        from meridian.cli.services import cmd_stop
        cmd_stop(args)

    elif args.command == "status":
        from meridian.cli.services import cmd_status
        cmd_status(args)

    elif args.command == "parse":
        from meridian.cli.parse import cmd_parse
        cmd_parse(args)

    elif args.command == "serve":
        from meridian.cli.serve import cmd_serve
        cmd_serve(args)

    elif args.command == "batch":
        from meridian.cli.batch import handle_batch_command
        handle_batch_command(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
