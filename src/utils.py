from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def time_block(label: str) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed = end - start
        print(f"[TIMER] {label}: {elapsed:.2f}s")
