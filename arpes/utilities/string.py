"""Some very simple string manipulation utilities."""

__all__ = [
    "safe_decode",
]


def safe_decode(input_bytes: bytes, prefer: str = "") -> str | None:
    """Tries different byte interpretations for decoding... very lazy."""
    codecs = ["utf-8", "latin-1", "ascii"]

    if prefer:
        codecs = [prefer, *codecs]

    for codec in codecs:
        try:
            return input_bytes.decode(codec)
        except UnicodeDecodeError:
            pass

    input_bytes.decode("utf-8")  # COULD NOT DETERMINE CODEC, RAISE
    return None
