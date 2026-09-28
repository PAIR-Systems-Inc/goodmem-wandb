"""Safe construction of GoodMem metadata filter expressions.

GoodMem filters are expression strings evaluated server-side. Building one by
interpolating caller data straight into the string is an injection risk, so
these helpers quote values and refuse field names that cannot be expressed
safely.

``val()`` returns JSON, so a comparison needs an explicit cast, and the cast
has to match the stored type: ``TEXT`` for strings, ``NUMERIC`` for numbers,
``BOOLEAN`` for booleans. :func:`from_mapping` picks the cast from the Python
value. A mismatched cast is not an error -- live (v1.0.320), metadata
``{"flag": true}`` filtered by ``CAST(val('$.flag') AS TEXT) = 'True'`` (or
``'true'``) is accepted with HTTP 200 and matches nothing, which looks exactly
like "nothing stored"; ``CAST(val('$.flag') AS BOOLEAN) = true`` matches.

Escaping was established against a live server (v1.0.320), not assumed. The
grammar escapes with a backslash: SQL-style ``''`` doubling and double-quoted
strings are both rejected with HTTP 400, and a raw newline inside a literal is
rejected outright.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import math
import re
from typing import Any

# A JSONPath member we are willing to build without escaping games.
_SAFE_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The filter grammar has no encoding for these inside a literal.
_FORBIDDEN_IN_LITERAL = re.compile(r"[\x00-\x1f\x7f]")


def _quote(value: str) -> str:
    """Single-quote a literal, backslash-escaping backslashes and quotes.

    Order matters: backslashes are escaped first so the backslash introduced
    for a quote is not escaped a second time.
    """
    if _FORBIDDEN_IN_LITERAL.search(value):
        raise ValueError(
            "Metadata filter values cannot contain control characters; the "
            "GoodMem filter grammar rejects them."
        )
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _check_field(field: str) -> str:
    if not _SAFE_FIELD.match(field):
        raise ValueError(
            f"Unsupported metadata field name {field!r}. Use letters, digits "
            "and underscores, or pass a filter expression directly."
        )
    return field


def _number(value: float) -> str:
    """Render a finite number as a plain decimal literal.

    ``repr()`` would produce ``nan``, ``inf`` or exponent notation such as
    ``1e+20``; the grammar has no literal for the first two.
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Metadata filter numbers must be finite.")
        return format(Decimal(repr(value)), "f")
    return str(value)


def _typed_literal(field: str, value: Any) -> tuple[str, str]:
    """The cast and the literal for one metadata value, chosen by its type."""
    # bool before int: bool is a subclass of int, and a boolean compared as a
    # number or as text matches nothing.
    if isinstance(value, bool):
        return "BOOLEAN", "true" if value else "false"
    if isinstance(value, (int, float)):
        return "NUMERIC", _number(value)
    if isinstance(value, str):
        return "TEXT", _quote(value)
    if value is None:
        raise ValueError(
            f"Metadata filter value for {field!r} is None, which has no filter "
            "equivalent. Use a str, int, float or bool, or pass a filter "
            "expression directly."
        )
    raise ValueError(
        f"Unsupported metadata filter value type {type(value).__name__} for "
        f"{field!r}; use str, int, float or bool."
    )


def text_equals(field: str, value: str) -> str:
    """Build an equality comparison against a text metadata field."""
    return f"CAST(val('$.{_check_field(field)}') AS TEXT) = {_quote(value)}"


def _equals(field: str, value: Any) -> str:
    """Equality cast to the type of ``value``: TEXT, NUMERIC or BOOLEAN."""
    cast, literal = _typed_literal(field, value)
    return f"CAST(val('$.{_check_field(field)}') AS {cast}) = {literal}"


def from_mapping(
    metadata_filter: Mapping[str, str | int | float | bool],
) -> str | None:
    """AND-join a mapping of field/value pairs into one filter expression.

    Each value is compared as its own type: a ``str`` as ``TEXT``, a ``bool``
    as ``BOOLEAN`` and an ``int`` or ``float`` as ``NUMERIC``, so
    ``{"category": "billing", "archived": False, "year": 2026}`` matches the
    text ``billing``, the boolean ``false`` and the number ``2026``. ``None``
    and any other type raise ``ValueError`` rather than being turned into
    text that silently matches nothing; so do non-finite numbers.

    Returns ``None`` for an empty mapping so callers can skip the filter.
    """
    clauses = [_equals(field, value) for field, value in metadata_filter.items()]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return " AND ".join(f"({clause})" for clause in clauses)


def combine(*expressions: str | None) -> str | None:
    """AND-join already-built expressions, ignoring ``None``."""
    present = [e for e in expressions if e]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return " AND ".join(f"({e})" for e in present)
