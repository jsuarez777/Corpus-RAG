"""PDF loaders, and the registry a config names one by.

``get_loader("pymupdf")`` is the swap point: a YAML config carries the name,
never the class, so changing extractor is a config edit.
"""

from app.rag.base import BaseLoader
from app.rag.loaders._pages import PAGE_SEPARATOR, join_pages, page_of
from app.rag.loaders.pymupdf_loader import PyMuPDFLoader
from app.rag.loaders.store import load_document, load_documents, save_document

# pdfplumber_loader lands here once its column/reflow handling is measured
# against this one.
LOADERS: dict[str, type[BaseLoader]] = {
    PyMuPDFLoader.name: PyMuPDFLoader,
}

DEFAULT_LOADER = PyMuPDFLoader.name


def get_loader(name: str = DEFAULT_LOADER, **kwargs) -> BaseLoader:
    """Build the loader registered under ``name``."""
    try:
        loader_cls = LOADERS[name]
    except KeyError:
        available = ", ".join(sorted(LOADERS))
        raise ValueError(f"Unknown loader {name!r}. Available: {available}") from None
    return loader_cls(**kwargs)


__all__ = [
    "DEFAULT_LOADER",
    "LOADERS",
    "PAGE_SEPARATOR",
    "BaseLoader",
    "PyMuPDFLoader",
    "get_loader",
    "join_pages",
    "load_document",
    "load_documents",
    "page_of",
    "save_document",
]
