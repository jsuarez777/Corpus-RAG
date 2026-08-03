"""Answer generation: numbered context in, cited answer out.

``get_llm()`` is the swap point, keyed by provider rather than by model — one
class serves every OpenAI model, and which model it runs is an argument
(``get_llm("openai", model="gpt-4.1-mini")``). That differs from the embedder
registry, where the alias *is* the model, because there the choice of model
changes the vectors and belongs in the result filename.
"""

from app.rag.base import BaseLLM
from app.rag.generation.citations import (
    DEFAULT_SNIPPET_CHARS,
    parse_citations,
    parse_markers,
    unresolved_markers,
)
from app.rag.generation.generator import DEFAULT_TOP_K, AnswerGenerator
from app.rag.generation.llm import DEFAULT_MODEL, DEFAULT_TEMPERATURE, OpenAILLM
from app.rag.generation.prompts import (
    DEFAULT_PASSAGE_CHARS,
    INSUFFICIENT,
    AnswerPrompt,
    build_context,
    latest_version,
    load_prompt,
)

LLMS: dict[str, type[BaseLLM]] = {OpenAILLM.name: OpenAILLM}

DEFAULT_LLM = OpenAILLM.name


def get_llm(name: str = DEFAULT_LLM, **kwargs) -> BaseLLM:
    """Build the LLM registered under ``name``.

    Constructing one does not call the API, but it does need the SDK installed
    and a key resolvable from the environment or ``~/.profile``.
    """
    try:
        llm_cls = LLMS[name]
    except KeyError:
        available = ", ".join(sorted(LLMS))
        raise ValueError(f"Unknown LLM {name!r}. Available: {available}") from None
    return llm_cls(**kwargs)


__all__ = [
    "DEFAULT_LLM",
    "DEFAULT_MODEL",
    "DEFAULT_PASSAGE_CHARS",
    "DEFAULT_SNIPPET_CHARS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "INSUFFICIENT",
    "LLMS",
    "AnswerGenerator",
    "AnswerPrompt",
    "BaseLLM",
    "OpenAILLM",
    "build_context",
    "get_llm",
    "latest_version",
    "load_prompt",
    "parse_citations",
    "parse_markers",
    "unresolved_markers",
]
