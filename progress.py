"""
Small progress-reporting helpers for training/inference logs.

Uses tqdm when available, with progress bars forced on even when stdout is
redirected to a file (e.g. a SLURM log). Falls back to throttled plain
prints when tqdm is not installed.
"""

import sys
import time

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except Exception:  # pragma: no cover - depends on environment
    _tqdm = None
    _HAS_TQDM = False


def phase(message):
    """Print a timestamped phase marker (always flushed)."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class _FallbackBar:
    """Throttled plain-print progress bar used when tqdm is unavailable."""

    def __init__(self, iterable, desc, total, unit):
        self._it = iter(iterable)
        self._desc = desc
        self._total = total
        self._unit = unit
        self._count = 0
        self._start = time.time()
        self._last_print = 0.0
        self._postfix = {}

    def set_postfix(self, **kwargs):
        self._postfix.update({k: str(v) for k, v in kwargs.items()})

    def set_description(self, desc):
        self._desc = desc

    def _maybe_print(self, force=False):
        now = time.time()
        if not force and now - self._last_print < 10.0:
            return
        self._last_print = now
        elapsed = now - self._start
        if self._total:
            frac = self._count / self._total
            rate = f"{elapsed / self._count:.2f}s/{self._unit}" if self._count else "-"
            eta = elapsed / self._count * (self._total - self._count) if self._count else 0.0
            eta_s = f"ETA {eta:.0f}s"
        else:
            frac = float("nan")
            rate = f"{self._count} {self._unit}"
            eta_s = ""
        post = ""
        if self._postfix:
            post = "  " + "  ".join(f"{k}={v}" for k, v in self._postfix.items())
        print(
            f"  {self._desc}: {self._count}/{self._total} ({frac:.1%})  "
            f"{rate}  {eta_s}{post}",
            flush=True,
        )

    def __iter__(self):
        for item in self._it:
            yield item
            self._count += 1
            self._maybe_print()

    def close(self):
        self._maybe_print(force=True)


class Progress:
    """tqdm-compatible subset: iterable wrapper + set_postfix/set_description."""

    def __init__(self, iterable, desc, total=None, unit="it"):
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)
        if _HAS_TQDM:
            self._bar = _tqdm(
                iterable,
                desc=desc,
                total=total,
                unit=unit,
                disable=False,     # keep the bar even when stdout is a log file
                file=sys.stdout,
                mininterval=5.0,   # avoid flooding the log file
            )
        else:
            self._bar = _FallbackBar(iterable, desc, total, unit)

    def __iter__(self):
        return iter(self._bar)

    def set_postfix(self, **kwargs):
        self._bar.set_postfix(**kwargs)

    def set_description(self, desc):
        self._bar.set_description(desc)

    def close(self):
        self._bar.close()


def progress_iter(iterable, desc, total=None, unit="it"):
    return Progress(iterable, desc=desc, total=total, unit=unit)
