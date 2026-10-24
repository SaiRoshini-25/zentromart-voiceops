
"""Safe adapters for the dataset's identity-gated offline tools."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tools import create_ticket, lookup_order  # noqa: E402

__all__ = ["lookup_order", "create_ticket"]
