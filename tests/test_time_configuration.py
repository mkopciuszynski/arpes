"""test for time configuration."""
from __future__ import annotations

import os.path
from typing import TYPE_CHECKING

import arpes.config

if TYPE_CHECKING:
    from collections.abc import Generator

    from .conftest import Sandbox


def test_patched_config(
    sandbox_configuration: Generator[Sandbox, None, None],
) -> None:
    """[TODO:summary].

    [TODO:description]

    Args:
        sandbox_configuration ([TODO:type]): [TODO:description]

    Returns:
        [TODO:description]
    """
    sandbox_configuration.with_workspace("basic")
    assert "name" in arpes.config.CONFIG["WORKSPACE"]
    assert arpes.config.CONFIG["WORKSPACE"]["name"] == "basic"
    assert "path" in arpes.config.CONFIG["WORKSPACE"]
    assert str(arpes.config.CONFIG["WORKSPACE"]["path"]).split(os.sep)[-2:] == ["datasets", "basic"]


def test_patched_config_no_workspace(
    sandbox_configuration: Generator[Sandbox, None, None],  # noqa: ARG001
) -> None:
    """[TODO:summary].

    [TODO:description]

    Args:
        sandbox_configuration ([TODO:type]): [TODO:description]

    Returns:
        [TODO:description]
    """
    assert arpes.config.CONFIG["WORKSPACE"] is None
