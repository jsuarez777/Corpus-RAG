"""Page-offset bookkeeping shared by every loader.

A ``Document`` carries one flat string, but qrels, citations and the UI all
want page numbers. Loaders record the char offset each page started at, and
:func:`page_of` turns any offset back into a 1-indexed page number.

A list of ints rather than a closure over the page list: it serializes, so the
map survives the JSON round-trip through ``data/extracted/``, and the lookup is
O(log n) instead of a scan per chunk.
"""

from bisect import bisect_right

# Blank line between pages: paragraph-reflowed text uses blank lines as
# paragraph breaks, so a page boundary reads as one to the segmenter too.
PAGE_SEPARATOR = "\n\n"


def join_pages(page_texts: list[str], separator: str = PAGE_SEPARATOR) -> tuple[str, list[int]]:
    """Join per-page text into one string plus the offset each page began at.

    ``page_starts[i]`` is the index in the returned string where page ``i + 1``
    starts. An empty page shares its successor's offset, which is why lookups
    use ``bisect_right``: an offset lands on the last page that could own it.
    """
    starts, cursor = [], 0
    for text in page_texts:
        starts.append(cursor)
        cursor += len(text) + len(separator)
    return separator.join(page_texts), starts


def page_of(page_starts: list[int], char_pos: int) -> int | None:
    """1-indexed page containing ``char_pos``, or None if no page map exists.

    Offsets past the end of the content resolve to the final page rather than
    raising — a chunk's ``end_char`` is exclusive and may sit one past it.
    """
    if not page_starts:
        return None
    return bisect_right(page_starts, char_pos)
