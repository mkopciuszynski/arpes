"""test for time configuration."""
import os.path

import arpes.config


def test_patched_config(sandbox_configuration) -> None:
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


def test_patched_config_no_workspace(sandbox_configuration) -> None:
    """[TODO:summary].

    [TODO:description]

    Args:
        sandbox_configuration ([TODO:type]): [TODO:description]

    Returns:
        [TODO:description]
    """
    assert arpes.config.CONFIG["WORKSPACE"] is None
