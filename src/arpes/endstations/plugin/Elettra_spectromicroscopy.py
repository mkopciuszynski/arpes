"""Implements data loading for the spectromicroscopy beamline at Elettra."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import h5py
import numpy as np
import xarray as xr

from arpes.endstations import HemisphericalEndstation, SynchrotronEndstation
from arpes.utilities import unwrap_xarray_item
from arpes import DATA_PATH

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes.endstations import ScanDesc

__all__ = ("SpectromicroscopyElettraEndstation",)


def collect_coord(index: int, dset: h5py.Dataset) -> tuple[str, NDArray[np.float64]]:
    """Uses the beamline metadata to normalize the coordinate information for a given axis.

    Args:
        index: The index of the coordinate to extract from metadata.
        dset: The HDF dataset containing Elettra spectromicroscopy data.

    Returns:
        The coordinate extracted at `index` from the metadata. The return convention here is to
        provide a tuple consisting of the extracted coordinate name, and the values for that
        coordinate.
    """
    shape = dset.shape
    name = dset.attrs[f"Dim{index} Name Units"][0].decode()
    start, delta = dset.attrs[f"Dim{index} Values"]
    num = shape[index]
    coords = np.linspace(start, start + delta * (num - 1), num)
    if name == "P":
        name = "phi"
    return name, coords


def h5_dataset_to_dataarray(dset: h5py.Dataset) -> xr.DataArray:
    flat_coords = [collect_coord(i, dset) for i in range(len(dset.shape))]

    def unwrap_bytestring(
        possibly_bytestring: bytes | list | tuple | NDArray[np.float64],
    ) -> str | list:
        if isinstance(possibly_bytestring, bytes):
            return possibly_bytestring.decode()

        if isinstance(possibly_bytestring, list | tuple | np.ndarray):
            return [unwrap_bytestring(elem) for elem in possibly_bytestring]

        return possibly_bytestring

    DROP_KEYS = {
        "Dim0 Name Units",
        "Dim1 Name Units",
        "Dim2 Name Units",
        "Dim3 Name Units",
        "Dim0 Values",
        "Dim1 Values",
        "Dim2 Values",
        "Dim3 Values",
    }

    coords = dict(flat_coords)
    attrs = {k: unwrap_bytestring(v) for k, v in dset.attrs.items() if k not in DROP_KEYS}

    # attr normalization
    attrs["T"] = round(attrs["Angular Coord"][0], 1)
    attrs["P"] = attrs["Angular Coord"][1]

    coords["P"] = attrs["P"]

    del attrs["Angular Coord"]  # temp
    del attrs["Date Time Start Stop"]  # temp
    del attrs["Temperature (K)"]  # temp
    del attrs["DET Limits"]  # temp
    del attrs["Energy Window (eV)"]  # temp
    del attrs["Ring Current (mA)"]  # temp
    del attrs["Stage Coord (XYZR)"]  # temp

    ring_info = attrs.pop("Ring En (GeV) GAP (mm) Photon (eV)", None)
    if False:  # <- not trustworthy info, try to autodetect the photon energy
        if isinstance(ring_info, list):
            en, gap, hv = ring_info
        else:
            ring_info = "".join(c for c in ring_info if c not in {"[", "]"})
            en, gap, hv = (float(item.strip()) for item in ring_info.split(","))

        attrs["hv"] = hv
        coords["hv"] = hv
        attrs["undulator_gap"] = gap
        attrs["ring_energy"] = en

    return xr.DataArray(
        dset[:],
        coords=coords,
        dims=[flat_coord[0] for flat_coord in flat_coords],
        attrs=attrs,
    )


class SpectromicroscopyElettraEndstation(
    HemisphericalEndstation,
    SynchrotronEndstation,
):
    """Data loading for the nano-ARPES beamline "Spectromicroscopy Elettra".

    Information available on the beamline can be accessed
    `here <https://www.elettra.trieste.it/elettra-beamlines/spectromicroscopy>`_.
    """

    PRINCIPAL_NAME = "Spectromicroscopy Elettra"
    ALIASES: ClassVar[list[str]] = ["Spectromicroscopy", "nano-ARPES Elettra"]

    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {
        ".hdf5",
    }
    _SEARCH_PATTERNS = (
        r"[\-a-zA-Z0-9_\w]+_[0]+{}$",
        r"[\-a-zA-Z0-9_\w]+_{}$",
        r"[\-a-zA-Z0-9_\w]+{}$",
        r"[\-a-zA-Z0-9_\w]+[0]{}$",
        r"{}" + (r"\\" if os.path.sep == "\\" else "/") + r"[\-a-zA-Z0-9_\w]+_001$",
    )

    @classmethod
    def files_for_search(cls: type, directory: str | Path) -> list[Path]:
        """Determines which files should be considered as candidates.

        Spectromicroscopy Elettra uses directories to group associated files together, so we have
        to find those.
        """
        base_files: list[Path] = []
        for file in Path(directory).iterdir():
            p = Path(directory) / file
            if p.is_dir():
                base_files = base_files + [Path(file) / f for f in p.iterdir()]
            else:
                base_files = [*base_files, Path(file)]

        return list(
            filter(
                lambda f: Path(f).suffix in cls._TOLERATED_EXTENSIONS,
                base_files,
            ),
        )

    ANALYZER_INFORMATION: ClassVar[dict[str, str | float | bool]] = {
        "analyzer": "Custom: in vacuum hemispherical",
        "analyzer_name": "Spectromicroscopy analyzer",
        "parallel_deflectors": False,
        "perpendicular_deflectors": False,
        "analyzer_radius": np.nan,
        "analyzer_type": "hemispherical",
    }

    RENAME_COORDS: ClassVar[dict[str, str]] = {
        "KE": "eV",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "P": "psi",
        "Angle": "phi",
    }
    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "Ep (eV)": "pass_energy",
        "Dwell Time (s)": "dwell_time",
        "Lens Mode": "lens_mode",
        "MCP Voltage": "mcp_voltage",
        "N of Scans": "n_scans",
        "Pressure (mbar)": "pressure",
        "Ring Current (mA)": "ring_current",
        # 'Ring En (GeV) Gap (mm) Photon(eV)': None,
        "Sample ID": "sample",
        "Stage Coord (XYZR)": "stage_coords",
        "Temperature (K)": "temperature",
    }

    CONCAT_COORDS: ClassVar[list[str]] = ["T", "P"]

    def concatenate_frames(
        self,
        frames: list[xr.Dataset],
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Concatenates frame for spectromicroscopy at Elettra.

        The scan axis is determined dynamically by checking for uniqueness across
        frames. The truth here is a bit more complicated because Elettra supports "diagonal" scans
        but frequently users set a very small offset in the other angular coordinate.
        """
        if scan_desc:
            warnings.warn("scan_desc is not supported", stacklevel=2)
        if not frames:
            msg = "Could not read any frames."
            raise ValueError(msg)

        if len(frames) == 1:
            return frames[0]

        # determine which axis to stitch them together along, and then do this
        scan_coord = None
        max_different_values = -np.inf
        best_coordinates = []

        for possible_scan_coord in self.CONCAT_COORDS:
            coordinates = [f.coords.get(possible_scan_coord, None) for f in frames]
            coordinates = [
                None if hasattr(c, "shape") and len(c.shape) else unwrap_xarray_item(c)
                for c in coordinates
            ]

            n_different_values = len(set(coordinates))
            if n_different_values > max_different_values and None not in coordinates:
                max_different_values = n_different_values
                scan_coord = possible_scan_coord
                best_coordinates = coordinates

        assert scan_coord is not None

        fs = []
        for c, f in zip(best_coordinates, frames, strict=True):
            frame_spectrum = f.spectrum
            frame_spectrum.coords[scan_coord] = c
            fs.append(frame_spectrum)

        return xr.Dataset({"spectrum": xr.concat(fs, scan_coord)})

    def resolve_frame_locations(
        self,
        scan_desc: ScanDesc | None = None,
    ) -> list[Path]:
        """Determines all files associated with a given scan.

        This beamline saves several HDF files in scan associated folders, so this
        amounts to checking whether the scan is multi-file and associating sibling
        files if so.
        """
        if scan_desc is None:
            msg = "Must pass dictionary as file scan_desc to all endstation loading code."
            raise ValueError(
                msg,
            )
        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        assert original_data_loc is not None
        if not Path(original_data_loc).exists():
            assert DATA_PATH
            original_data_loc = Path(DATA_PATH) / original_data_loc
        p = Path(original_data_loc)
        if p.parent.parent.stem in ([*list(self._SEARCH_DIRECTORIES), "data"]):
            return list(p.parent.glob("*.hdf5"))
        return [p]

    def load_single_frame(
        self,
        frame_path: str | Path = "",
        scan_desc: ScanDesc | None = None,
        **kwargs: Incomplete,
    ) -> xr.Dataset:
        """Loads a single HDF file with spectromicroscopy Elettra data."""
        if scan_desc:
            warnings.warn("scan_desc is not supported.", stacklevel=2)
        if kwargs:
            warnings.warn("Any kwargs is not supported.", stacklevel=2)
        with h5py.File(frame_path, "r") as f:
            arrays = {k: h5_dataset_to_dataarray(f[k]) for k in f}

            if len(arrays) == 1:
                arrays = {"spectrum": next(iter(arrays.values()))}

            return xr.Dataset(arrays)

    def postprocess_final(
        self,
        data: xr.Dataset,
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Performs final postprocessing of the data.

        This mostly amounts to:
        1. Adjusting for the work function and converting kinetic to binding energy
        2. Adjusting angular coordinates to standard conventions
        3. Microns -> millimeters on spatial coordinates
        """
        data = data.rename({k: v for k, v in self.RENAME_COORDS.items() if k in data.coords})

        if "eV" in data.coords:
            approx_workfunction = 3.46
            data.coords["hv"] = 27.0 if data.eV.mean().item() < 29 else 74.0  # noqa: PLR2004
            data.eV.values += approx_workfunction - data.coords["hv"].item()

        for coord, default in {"psi": 90.0, "phi": 0.0}.items():
            if coord not in data.coords:
                data.coords[coord] = default

        data.coords["psi"] = np.deg2rad(data.coords["psi"] - 90)
        data.coords["psi"] = np.deg2rad(data.coords["psi"] - 90)
        data.coords["phi"] = np.deg2rad(data.coords["phi"] + data.spectrum.attrs["T"])
        data.coords["beta"] = 0.0
        data.coords["chi"] = 0.0
        data.coords["alpha"] = np.pi / 2
        data.coords["theta"] = 0.0

        for i, dim_name in enumerate(["x", "y", "z"]):
            if dim_name in data.coords:
                data.coords[dim_name] = data.coords[dim_name] / 1000.0
            else:
                try:
                    data.coords[dim_name] = [
                        dv for dv in data.data_vars.values() if "eV" in dv.dims
                    ][0].attrs["stage_coords"][i] / 1000.0
                except IndexError:
                    data.coords[dim_name] = 0.0

        return super().postprocess_final(data, scan_desc)
