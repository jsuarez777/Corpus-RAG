"""Vector stores, and the registry a config names one by."""

from app.rag.base import BaseVectorStore
from app.rag.stores.faiss_store import FaissStore, unit_rows
from app.rag.stores.store import (
    config_id,
    index_chunk_set_id,
    index_dir,
    require_same_chunk_set,
)

STORES: dict[str, type[BaseVectorStore]] = {
    FaissStore.name: FaissStore,
}

DEFAULT_STORE = FaissStore.name


def get_store(name: str = DEFAULT_STORE, **kwargs) -> BaseVectorStore:
    """Build the store registered under ``name``."""
    try:
        store_cls = STORES[name]
    except KeyError:
        available = ", ".join(sorted(STORES))
        raise ValueError(f"Unknown store {name!r}. Available: {available}") from None
    return store_cls(**kwargs)


def open_store(path, name: str = DEFAULT_STORE) -> BaseVectorStore:
    """Build a store and restore a saved index into it, in one step."""
    store = get_store(name)
    store.load(path)
    return store


__all__ = [
    "DEFAULT_STORE",
    "STORES",
    "BaseVectorStore",
    "FaissStore",
    "config_id",
    "get_store",
    "index_chunk_set_id",
    "index_dir",
    "open_store",
    "require_same_chunk_set",
    "unit_rows",
]
