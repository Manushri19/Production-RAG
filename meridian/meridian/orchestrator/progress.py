"""
Progress Display Utilities

Functions for displaying batch progress in the terminal.
"""

from typing import List, Dict, Any


def format_duration(seconds: int) -> str:
    """Format seconds as human-readable duration."""
    if seconds is None:
        return "N/A"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def progress_bar(percent: float, width: int = 40) -> str:
    """Generate a text progress bar."""
    filled = int(width * percent / 100)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


def print_progress(progress: Dict[str, Any]):
    """Print detailed progress for a batch."""
    batch_id = progress.get("batch_id", "unknown")
    status = progress.get("status", "unknown").upper()

    total = progress.get("total", 0)
    pending = progress.get("pending", 0)
    processing = progress.get("processing", 0)
    complete = progress.get("complete", 0)
    failed = progress.get("failed", 0)
    percent = progress.get("percent_complete", 0)

    elapsed = progress.get("elapsed_seconds", 0)
    docs_per_min = progress.get("docs_per_minute", 0)
    remaining = progress.get("estimated_remaining_seconds")
    timing_summary = progress.get("timing_summary", {})

    print()
    print("=" * 70)
    print(f"BATCH: {batch_id}")
    print("=" * 70)
    print()
    print(f"  Status:     {status}")
    print(f"  Collection: {progress.get('collection', 'N/A')}")
    print(f"  Input:      {progress.get('input_dir', 'N/A')}")
    print()
    print(f"  Progress:   {progress_bar(percent)} {percent:.1f}%")
    print()
    print(f"  Documents:")
    print(f"    Total:      {total:,}")
    print(f"    Complete:   {complete:,} ({complete/total*100:.1f}%)" if total > 0 else f"    Complete:   {complete:,}")
    print(f"    Processing: {processing:,}")
    print(f"    Pending:    {pending:,}")
    print(f"    Failed:     {failed:,} ({failed/total*100:.1f}%)" if total > 0 and failed > 0 else f"    Failed:     {failed:,}")
    print()

    # Show accurate timing if we have per-doc timing data
    if timing_summary.get("count", 0) > 0:
        print(f"  Timing (actual processing):")
        wall_clock = timing_summary.get("wall_clock_seconds")
        if wall_clock:
            print(f"    Wall clock: {format_duration(int(wall_clock))}")
        print(f"    Per doc:    min={timing_summary.get('min_seconds', 0):.1f}s, max={timing_summary.get('max_seconds', 0):.1f}s, avg={timing_summary.get('avg_seconds', 0):.1f}s")
        print(f"    Total CPU:  {format_duration(int(timing_summary.get('total_seconds', 0)))}")
        if wall_clock and wall_clock > 0:
            actual_throughput = timing_summary.get("count", 0) / (wall_clock / 60)
            print(f"    Throughput: {actual_throughput:.1f} docs/min")

        # Page stats
        total_pages = timing_summary.get("total_pages", 0)
        if total_pages > 0:
            print()
            print(f"  Pages:")
            print(f"    Total:      {total_pages:,} pages")
            if wall_clock and wall_clock > 0:
                pages_per_min = total_pages / (wall_clock / 60)
                print(f"    Throughput: {pages_per_min:.1f} pages/min")
            if timing_summary.get("avg_sec_per_page"):
                print(f"    Per page:   min={timing_summary.get('min_sec_per_page', 0):.2f}s, max={timing_summary.get('max_sec_per_page', 0):.2f}s, avg={timing_summary.get('avg_sec_per_page', 0):.2f}s")
    else:
        print(f"  Timing:")
        print(f"    Elapsed:    {format_duration(elapsed)}")
        print(f"    Throughput: {docs_per_min:.1f} docs/min")
        if remaining and remaining > 0:
            print(f"    Remaining:  ~{format_duration(remaining)}")
    print()
    print("=" * 70)


def print_batch_list(batches: List[Dict[str, Any]]):
    """Print summary list of batches."""
    print()
    print("=" * 70)
    print("BATCHES")
    print("=" * 70)
    print()
    print(f"{'BATCH ID':<30} {'STATUS':<10} {'PROGRESS':<20} {'FAILED':<8}")
    print("-" * 70)

    for batch in batches:
        batch_id = batch.get("batch_id", "unknown")
        status = batch.get("status", "?")
        total = batch.get("total", 0)
        complete = batch.get("complete", 0)
        failed = batch.get("failed", 0)
        percent = batch.get("percent_complete", 0)

        progress_str = f"{complete}/{total} ({percent:.0f}%)"
        failed_str = str(failed) if failed > 0 else "-"

        print(f"{batch_id:<30} {status:<10} {progress_str:<20} {failed_str:<8}")

    print()


def print_failed_docs(failed: List[Dict[str, str]], limit: int = 20):
    """Print list of failed documents."""
    print()
    print("=" * 70)
    print(f"FAILED DOCUMENTS ({len(failed)} total)")
    print("=" * 70)
    print()

    for i, doc in enumerate(failed[:limit]):
        doc_id = doc.get("doc_id", "unknown")
        error = doc.get("error", "Unknown error")

        # Truncate error for display
        if len(error) > 60:
            error = error[:57] + "..."

        print(f"  {i+1}. {doc_id}")
        print(f"     Error: {error}")
        print()

    if len(failed) > limit:
        print(f"  ... and {len(failed) - limit} more")
        print()

    print("=" * 70)
