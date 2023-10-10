"""Utilities for modifying, iterating over, and transforming dictionaries."""
from __future__ import annotations

import re
from typing import Any

from arpes.utilities.xarray import lift_dataarray_attrs, lift_datavar_attrs

__all__ = (
    "rename_keys",
    "clean_keys",
    "rename_dataarray_attrs",
    "clean_datavar_attribute_names",
    "clean_attribute_names",
    "case_insensitive_get",
)


def _rename_key(
    d: dict[str, Any],
    original_name_k: str,
    new_name_k: str,
) -> None:
    if original_name_k in d:
        d[new_name_k] = d[original_name_k]
        del d[original_name_k]


def rename_keys(
    d: dict[str, Any],
    keys_dict: dict[str, str],
) -> dict[str, Any]:
    """Renames all the keys of `d` according to the remapping in `keys_dict`.

    Args:
        d (dict): dict object (Suppose the attrs)
        keys_dict(dict[str, str]):  {original_name_k: new_name_k}

    Returns:
        [TODO:description]
    """
    d = d.copy()
    for k, nk in keys_dict.items():
        _rename_key(d, k, nk)

    return d


def clean_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Renames dictionary keys so that they are more Pythonic."""

    def clean_single_key(k: str) -> str:
        k = k.replace(" ", "_")
        k = k.replace(".", "_")
        k = k.lower()
        k = re.sub(r"[()/?]", "", k)
        return k.replace("__", "_")

    return dict(zip([clean_single_key(k) for k in d], d.values(), strict=True))


def case_insensitive_get(
    d: dict[str, object],
    key: str,
    default: object = None,
    *,
    take_first: bool = False,
) -> object:
    """Looks up a key in a dictionary ignoring case.

    We use this sometimes to be nicer to users who don't provide perfectly sanitized data.

    Args:
        d: The dictionary to perform lookup in
        key: The key to get
        default: A default value if the key is not present
        take_first: Whether to take the first entry if there were multiple found
    """
    found_value = False
    value = None

    for k, v in d.items():
        if k.lower() == key.lower():
            if not take_first and found_value:
                msg = "Duplicate case insensitive keys"
                raise ValueError(msg)

            value = v
            found_value = True

            if take_first:
                break

    if not found_value:
        return default

    return value


rename_dataarray_attrs = lift_dataarray_attrs(rename_keys)
clean_attribute_names = lift_dataarray_attrs(clean_keys)

clean_datavar_attribute_names = lift_datavar_attrs(clean_keys)
