"""Provides cached data fixtures for tests."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from arpes.io import load_data

if TYPE_CHECKING:
    import xarray as xr

__all__ = ["cache_loader"]


TEST_ROOT = (Path(__file__).parent).absolute()


def path_to_datasets() -> Path:
    return TEST_ROOT / "resources" / "datasets"


@dataclass
class CachingDataLoader:
    cache: dict[str, xr.Dataset] = field(default_factory=dict)

    def load_test_scan(self, example_name: str | Path, **kwargs) -> xr.Dataset:
        """[TODO:summary].

        [TODO:description]

        Args:
            example_name ([TODO:type]): [TODO:description]
            kwargs: Pass to load_data function

        Raises:
            ValueError: [TODO:description]
        """
        if example_name in self.cache:
            return self.cache[str(example_name)].copy(deep=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path_to_data = path_to_datasets() / example_name
            if not path_to_data.exists():
                msg = f"{path_to_data!s} does not exist."
                raise ValueError(msg)

            data = load_data(str(path_to_data.absolute()), **kwargs)
            self.cache[example_name] = data
            return data.copy(deep=True)


cache_loader = CachingDataLoader()
