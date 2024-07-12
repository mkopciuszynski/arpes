"""Top level module for PyARPES."""

# pylint: disable=unused-import
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import ConfigSettings, ConfigType
# Use both version conventions for people's sanity.
VERSION = "4.0.0"
__version__ = VERSION


__all__ = ["__version__"]


SOURCE_ROOT = str(Path(__file__).parent)
DATA_PATH: str | None = None

SETTINGS: ConfigSettings = {
    "interactive": {
        "main_width": 350,
        "marginal_width": 150,
        "palette": "magma",
    },
    "use_tex": False,
}
CONFIG: ConfigType = {
    "WORKSPACE": {},
    "CURRENT_CONTEXT": None,
    "ENABLE_LOGGING": True,
    "LOGGING_STARTED": False,
    "LOGGING_FILE": None,
}
