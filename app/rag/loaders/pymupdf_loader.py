"""PyMuPDF loader — the default extractor.

Fast (tens of pages per second) and dependency-light, which is what makes it
the right default for the experiment grid.

**``sort_blocks`` defaults to False, and on this corpus that is load-bearing.**
``get_text(sort=True)`` orders blocks by position on the page, which is the
right repair for a single-column PDF whose content stream is scrambled — and
precisely the wrong thing for a two-column paper, where "next block down the
page" alternates between the two columns. It weaves them together mid-sentence
and produces text that was never written. LaTeX-produced arXiv PDFs declare
their blocks in reading order already, so the untouched order is the good one.

What remains after that is line wrapping: PDF has no paragraphs, so a sentence
typeset over four lines arrives with three newlines in it. Rejoining those is
paragraph reflow, and it belongs in ``preprocess``.
"""

from collections.abc import Callable
from pathlib import Path

import pymupdf

from app.rag.base import BaseLoader
from app.rag.loaders._pages import PAGE_SEPARATOR, join_pages
from app.rag.models import Document, DocumentMetadata
from app.rag.preprocessing import DEFAULT_PREPROCESSOR, resolve_preprocessor

# PyMuPDF block tuples are (x0, y0, x1, y1, text, block_no, block_type);
# block_type 1 is an image, which has no text worth keeping.
_BLOCK_TEXT, _BLOCK_TYPE = 4, 6
_TEXT_BLOCK = 0


class PyMuPDFLoader(BaseLoader):
    """Extracts a PDF's text with PyMuPDF, one Document per file.

    ``preprocess`` runs per page before the pages are joined, so ``page_starts``
    is taken from the final text — never clean a ``Document`` afterwards.
    It takes a registry name (``"default"``, ``"reflow"``, ``"none"``) so a
    config can select one, or a callable.

    ``layout="blocks"`` extracts PyMuPDF's block structure and separates blocks
    with a blank line, which is what gives the reflow step paragraph boundaries
    to respect; ``layout="text"`` is the raw stream, kept for comparison.
    """

    name = "pymupdf"

    def __init__(
        self,
        *,
        preprocess: str | Callable[[str], str] | None = DEFAULT_PREPROCESSOR,
        page_separator: str = PAGE_SEPARATOR,
        sort_blocks: bool = False,
        layout: str = "blocks",
    ) -> None:
        if layout not in {"blocks", "text"}:
            raise ValueError(f"layout must be 'blocks' or 'text', got {layout!r}")
        self._preprocess = resolve_preprocessor(preprocess)
        self._page_separator = page_separator
        self._sort_blocks = sort_blocks
        self._layout = layout

    def _page_text(self, page) -> str:
        """One page's text, with a blank line between paragraph blocks."""
        if self._layout == "text":
            return page.get_text("text", sort=self._sort_blocks)
        blocks = page.get_text("blocks", sort=self._sort_blocks)
        return "\n\n".join(
            block[_BLOCK_TEXT].strip()
            for block in blocks
            if block[_BLOCK_TYPE] == _TEXT_BLOCK and block[_BLOCK_TEXT].strip()
        )

    def load(self, path: Path) -> Document:
        """Extract ``path`` into a Document with page offsets and a paper id."""
        path = Path(path)
        with pymupdf.open(path) as pdf:
            raw_metadata = pdf.metadata or {}
            page_texts = [self._page_text(page) for page in pdf]

        if self._preprocess is not None:
            page_texts = [self._preprocess(text) for text in page_texts]

        content, page_starts = join_pages(page_texts, self._page_separator)

        return Document(
            content=content,
            metadata=DocumentMetadata(
                source=path.name,
                title=_clean(raw_metadata.get("title")),
                author=_clean(raw_metadata.get("author")),
                page_count=len(page_texts),
                # Extras (DocumentMetadata is extra="allow"): the page map that
                # chunk offsets resolve against, and the arXiv id that qrels
                # keys on. document.id stays the in-process UUID link.
                page_starts=page_starts,
                paper_id=path.stem,
                loader=self.name,
                # Extraction settings ride along, so a cached Document in
                # data/extracted/ says how it was produced.
                layout=self._layout,
                preprocess=getattr(self._preprocess, "__name__", None),
            ),
        )


def _clean(value: str | None) -> str | None:
    """PyMuPDF reports absent metadata as an empty string; None reads better."""
    value = (value or "").strip()
    return value or None
