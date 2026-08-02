"""Dense embeddings from sentence-transformers.

Returns model output as-is; this class adds no normalization of its own. Unit
length is a requirement of the metric the store chooses — ``IndexFlatIP`` needs
it, an L2-distance index would not — so
:class:`~app.rag.stores.faiss_store.FaissStore` normalizes on the way in and on
every query. Normalizing here as well would quietly assert which store this
embedder gets paired with, and in the experiment grid the two vary
independently.

As it happens both grid models ship their own ``Normalize`` layer, so the
store's pass is a no-op for them. That is the argument for the arrangement
rather than against it: the store does not have to know which models normalize
themselves, and one that does not (bge, e5, a bare transformer) still yields
cosine scores.

The model is loaded lazily. Constructing an embedder is cheap enough to do in
``--help`` paths and registry sweeps; the first ``embed_texts`` call is what
pays the download and the several hundred MB of weights.
"""

from __future__ import annotations

import logging
from functools import cached_property

from app.rag.base import BaseEmbedder

log = logging.getLogger(__name__)

# Short names for the two models the grid compares, so a config carries
# `minilm` rather than a HuggingFace path. Any other string is passed through
# to sentence-transformers untouched.
MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

DEFAULT_MODEL = "minilm"


class SentenceTransformerEmbedder(BaseEmbedder):
    """Encode text with a sentence-transformers bi-encoder.

    ``model`` is an alias from :data:`MODELS` or any model id the library
    accepts. ``device`` defaults to whatever sentence-transformers picks, which
    is CUDA or Apple MPS when present and CPU otherwise.
    """

    name = "sentence_transformer"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        batch_size: int = 64,
        device: str | None = None,
        show_progress: bool = False,
    ) -> None:
        self.alias = model
        self.model_id = MODELS.get(model, model)
        self.batch_size = batch_size
        self.device = device
        self.show_progress = show_progress

    @cached_property
    def _model(self):
        # Imported here, not at module scope: the registry imports this module
        # to list what is available, and that must not drag in torch.
        from sentence_transformers import SentenceTransformer

        log.info(f"Loading {self.model_id} (first use downloads it)")
        return SentenceTransformer(self.model_id, device=self.device)

    @property
    def dimension(self) -> int:
        """Vector length, taken from the model rather than a lookup table."""
        # Renamed in sentence-transformers 5; the old name still works but warns.
        getter = getattr(self._model, "get_embedding_dimension", None)
        return getter() if getter else self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``, one raw vector per input, in order."""
        if not texts:
            # encode([]) is not reliably shaped across versions, and an empty
            # corpus is a legitimate no-op rather than an error.
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.alias!r})"
