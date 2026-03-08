"""Centralized logging configuration for vectorscope."""

import json
import logging
import time


class _JSONLFormatter(logging.Formatter):
    """Format log records as JSONL lines for machine-readable perf log."""

    def format(self, record):
        data = record.msg if isinstance(record.msg, dict) else {'msg': str(record.msg)}
        data.setdefault('ts', time.strftime("%Y-%m-%d %H:%M:%S"))
        return json.dumps(data)


class _EventFormatter(logging.Formatter):
    """Format log records as timestamped structured text."""

    def format(self, record):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cat = getattr(record, 'category', '')
        if cat:
            return f"[{ts}] {cat:<8} {record.getMessage()}"
        return f"[{ts}] {record.getMessage()}"


class _CategoryFilter(logging.Filter):
    """Only pass records whose 'category' extra matches the allowed set."""

    def __init__(self, categories):
        super().__init__()
        self.categories = {c.strip().lower() for c in categories}

    def filter(self, record):
        return getattr(record, 'category', '') in self.categories


def setup_logging(log_categories=None):
    """Set up vectorscope loggers. Safe to call multiple times.

    log_categories: iterable of category strings to allow (e.g. ['level','spawn']).
                    None (default) means all categories pass through.
    """

    # Performance logger — JSONL
    perf = logging.getLogger('vectorscope.perf')
    perf.setLevel(logging.INFO)
    if not perf.handlers:
        h = logging.FileHandler("vectorscope_perf.log", encoding="utf-8")
        h.setFormatter(_JSONLFormatter())
        perf.addHandler(h)
        perf.propagate = False

    # Game logger — structured text
    game = logging.getLogger('vectorscope.game')
    game.setLevel(logging.DEBUG)
    if not game.handlers:
        h = logging.FileHandler("vectorscope_asteroids.log", encoding="utf-8")
        h.setFormatter(_EventFormatter())
        if log_categories is not None:
            h.addFilter(_CategoryFilter(log_categories))
        game.addHandler(h)
        game.propagate = False

    # Web logger — structured text
    web = logging.getLogger('vectorscope.web')
    web.setLevel(logging.INFO)
    if not web.handlers:
        h = logging.FileHandler("vectorscope_web.log", encoding="utf-8")
        h.setFormatter(_EventFormatter())
        web.addHandler(h)
        web.propagate = False
