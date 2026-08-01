"""Sliding window — fixed-size windows with a large, ratio-defined overlap.

The spec lists this as its own strategy, but algorithmically it *is*
``fixed_size``: a window of N tokens advancing by N-k. What differs is the
intent, so what differs here is the parameterization. Overlap is a **ratio**
of the window, because the property that matters — how many windows any given
sentence appears in — is the ratio, and a fixed token count silently changes
that whenever ``window_size`` moves.

Subclassing rather than duplicating: the window logic has exactly one
implementation, so an offset bug can only be fixed once.
"""

from __future__ import annotations

from app.rag.chunking.fixed_size import FixedSizeChunker


class SlidingWindowChunker(FixedSizeChunker):
    """Fixed windows whose overlap is a fraction of the window size."""

    name = "sliding_window"

    def __init__(self, window_size: int = 512, overlap_ratio: float = 0.5) -> None:
        if not 0.0 <= overlap_ratio < 1.0:
            # 1.0 would mean the window never advances.
            raise ValueError(f"overlap_ratio must be in [0.0, 1.0), got {overlap_ratio}")
        self.overlap_ratio = overlap_ratio
        super().__init__(chunk_size=window_size, overlap=int(window_size * overlap_ratio))

    @property
    def window_size(self) -> int:
        """Alias for ``chunk_size``, in this strategy's own vocabulary."""
        return self.chunk_size

    def __repr__(self) -> str:
        return f"SlidingWindowChunker(window_size={self.window_size}, overlap_ratio={self.overlap_ratio})"
