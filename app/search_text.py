import re

_TOKEN = re.compile(r"\{\{[pf]:\d+\}\}")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "and", "for", "with", "from", "into", "onto", "this", "that",
    "there", "their", "they", "them", "photo", "image", "picture", "person",
    "people", "show", "shows", "visible", "wearing", "standing", "sitting",
    "photos", "pictures", "images", "find", "search", "taken", "near", "around",
}


def build_search_text(caption="", activity="", scene="", tags=None):
    """Build a compact, name-free document for local text retrieval."""
    pieces = []
    for value in (caption, activity, scene, *(tags or [])):
        if not isinstance(value, str):
            continue
        value = _TOKEN.sub("person", value)
        value = re.sub(r"\bP\d+\b", "person", value, flags=re.IGNORECASE)
        value = " ".join(value.split()).strip(" .,:;-_")
        if value and value.casefold() not in {piece.casefold() for piece in pieces}:
            pieces.append(value)
    return ". ".join(pieces)


def search_terms(text):
    return {word for word in _WORD.findall((text or "").lower()) if len(word) > 2 and word not in _STOP}
