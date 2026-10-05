"""
pytest configuration for Oracle tests.
Sets up the oracle package on sys.path for import.
"""

import sys
from pathlib import Path

# Ensure the project root is on the path so `oracle` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
