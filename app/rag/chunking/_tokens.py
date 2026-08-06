"""Token counting and the token-index -> char-offset map.

Every chunker cuts on token counts but must report char offsets, and
:func:`token_starts` is the bridge — it is the one function the whole offset
discipline rests on.

The encoding is ``cl100k_base`` throughout, so a chunk's token count means the
same thing at chunking time as it does at embedding time.
"""

from __future__ import annotations

TOKEN_ENCODING = "cl100k_base"

_encoder = None


def get_encoder():
    """The shared tiktoken encoder, built on first use.

    Imported lazily: tiktoken downloads its vocabulary on first construction,
    and nothing that merely imports this module should pay for that.
    """
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding(TOKEN_ENCODING)
    return _encoder


def count_tokens(text: str) -> int:
    """Number of ``cl100k_base`` tokens in ``text``."""
    # disallowed_special=() so a document containing the literal text
    # "<|endoftext|>" counts as tokens instead of raising.
    return len(get_encoder().encode(text, disallowed_special=()))


def token_starts(text: str) -> tuple[list[int], list[int]]:
    """Return ``(tokens, starts)`` where ``starts[i]`` is the char offset token
    ``i`` begins at, and ``starts[len(tokens)] == len(text)``.

    Token byte sequences concatenate to exactly the UTF-8 encoding of the text,
    so cumulative token-byte lengths give byte offsets, which are then walked
    back to char offsets. A token beginning mid-character (a multi-byte char
    split across two tokens) maps to that character's own offset, so slicing at
    any ``starts[i]`` never cuts a character in half.
    """
    encoder = get_encoder()
    tokens = encoder.encode(text, disallowed_special=())

    byte_ends: list[int] = []  # byte offset just past character i
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        byte_ends.append(total)

    starts: list[int] = []
    char_index = 0
    byte_pos = 0
    for token in tokens:
        while char_index < len(byte_ends) and byte_ends[char_index] <= byte_pos:
            char_index += 1
        starts.append(char_index)
        byte_pos += len(encoder.decode_single_token_bytes(token))
    starts.append(len(text))
    return tokens, starts


def resolve_overlap(overlap: int | str, size: int) -> int:
    """Normalize an overlap given as tokens (``128``) or a percentage (``"25%"``).

    Percentages are the useful form for sliding-window presets, where the point
    is the *ratio* rather than a token count that changes meaning with ``size``.
    """
    if isinstance(overlap, str):
        text = overlap.strip()
        if not text.endswith("%"):
            raise ValueError(f"Overlap {overlap!r} must be an int or a percentage like '25%'")
        try:
            percent = float(text[:-1])
        except ValueError:
            raise ValueError(f"Invalid overlap percentage: {overlap!r}") from None
        overlap = int(size * percent / 100)

    if overlap < 0:
        raise ValueError(f"Overlap must be >= 0, got {overlap}")
    if overlap >= size:
        # Equal would mean zero forward progress; the loop would never end.
        raise ValueError(f"Overlap ({overlap}) must be less than chunk size ({size})")
    return overlap
