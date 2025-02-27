"""Utility functions for extracting ARPES information from the FITS file conventions."""

from __future__ import annotations

import functools
import warnings
from ast import literal_eval
from collections.abc import Iterable
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy import ndarray
from numpy._typing import NDArray

from arpes.debug import setup_logger
from arpes.utilities.funcutils import collect_leaves, iter_leaves

if TYPE_CHECKING:
    from astropy.io.fits.hdu.table import BinTableHDU

__all__ = (
    "extract_coords",
    "find_clean_coords",
)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


DEFAULT_DIMENSION_RENAMINGS: dict[str, str] = {
    "Beta": "beta",
    "Theta": "theta",
    "Delay": "delay",
    "Sample-X": "cycle",
    "null": "cycle",
    "Mira": "pump_power",
    "X": "x",
    "Y": "y",
    "Z": "z",
}

CoordsDict: TypeAlias = dict[str, NDArray[np.float64]]
Dimension = str


def extract_coords(
    attrs: dict[str, Any],
    dimension_renamings: dict[str, str] | None = None,
) -> tuple[CoordsDict, list[Dimension], list[int]]:
    """Does the hard work of extracting coordinates from the scan description.

    Args:
        attrs:
        dimension_renamings:

    Returns:
        A tuple consisting of the coordinate arrays, the dimension names, and their shapes
    """
    dimension_renamings = dimension_renamings or DEFAULT_DIMENSION_RENAMINGS

    n_loops = attrs["LWLVLPN"]
    if n_loops is None:
        return {}, [], []
    logger.debug(f"Found n_loops={n_loops}")
    scan_coords, scan_dimension, scan_shape = {}, [], []

    for loop in range(n_loops):
        n_scan_dimensions = attrs[f"NMSBDV{loop}"]
        logger.debug(f"Considering loop {loop}, n_scan_dimensions={n_scan_dimensions}")
        if attrs[f"SCNTYP{loop}"] == 0:
            logger.debug("Loop is computed")
            for i in range(n_scan_dimensions):
                name, start, end, n = (
                    attrs[f"NM_{loop}_{i}"],
                    float(attrs[f"ST_{loop}_{i}"]),
                    float(attrs[f"EN_{loop}_{i}"]),
                    int(attrs[f"N_{loop}_{i}"]),
                )

                name = dimension_renamings.get(name, name)

                scan_dimension.append(name)
                scan_shape.append(n)
                scan_coords[name] = np.linspace(start, end, n, endpoint=True)
            # else:  # tabulated scan, this is more complicated
            # In the past this branch has been especially tricky.
            # I know of at least two client pieces of data:
            #    * Tabulated scans which include angle-compensated scans
            #    * Multi region scans at MAESTRO
            #
            # Remarkably, I'm having a hard time figuring out how this code ever worked
            # in the past for beta compensated scans which appear to be stored with a literal table.
            # I think in the past I probably neglected to unfreeze the tabulated coordinates which
            # were attached since they do not matter much from the perspective of analysis.
            #
            # As of 2021, that is the perspective we are taking on the issue.
        elif n_scan_dimensions > 1:
            logger.debug("Loop is tabulated and is not region based")
            for i in range(n_scan_dimensions):
                name = attrs[f"NM_{loop}_{i}"]
                if f"ST_{loop}_{i}" not in attrs and f"PV_{loop}_{i}_0" in attrs:
                    msg = f"Determined that coordinate {name} "
                    msg += "is tabulated based on scan coordinate. Skipping!"
                    logger.debug(msg)
                    continue
                start, end, n = (
                    float(attrs[f"ST_{loop}_{i}"]),
                    float(attrs[f"EN_{loop}_{i}"]),
                    int(attrs[f"N_{loop}_{i}"]),
                )

                old_name = name
                name = dimension_renamings.get(name, name)
                logger.debug(f"Renaming: {old_name} -> {name}")

                scan_dimension.append(name)
                scan_shape.append(n)
                scan_coords[name] = np.linspace(start, end, n, endpoint=True)

        else:
            logger.debug("Loop is tabulated and is region based")
            name, n = (
                attrs[f"NM_{loop}_0"],
                attrs[f"NMPOS_{loop}"],
            )

            try:
                n_regions_key = {"Delay": "DS_NR"}.get(name, "DS_NR")
                n_regions = attrs[n_regions_key]

                name = dimension_renamings.get(name, name)
            except KeyError:
                if f"ST_{loop}_1" in attrs:
                    warnings.warn("More than one region detected but unhandled.", stacklevel=2)

                n_regions = 1
                name = dimension_renamings.get(name, name)

            logger.debug(f"Loop (name, n_regions, size) = {(name, n_regions, n)}")

            coord: NDArray[np.float64] = np.array(())
            for region in range(n_regions):
                start, end, n = (
                    attrs[f"ST_{loop}_{region}"],
                    attrs[f"EN_{loop}_{region}"],
                    attrs[f"N_{loop}_{region}"],
                )
                msg = f"Reading coordinate {region} from loop. (start, end, n)"
                msg += f"{(start, end, n)}"

                logger.debug(msg)

                coord = np.concatenate((coord, np.linspace(start, end, n, endpoint=True)))

            scan_dimension.append(name)
            scan_shape.append(len(coord))
            scan_coords[name] = coord
    return scan_coords, scan_dimension, scan_shape


def _handle_computed_loop(
    attrs, loop, n_scan_dimensions, scan_coords, scan_dimension, scan_shape, dimension_renamings
) -> None:
    logger.debug("Loop is computed")
    for i in range(n_scan_dimensions):
        name, start, end, n = (
            attrs[f"NM_{loop}_{i}"],
            float(attrs[f"ST_{loop}_{i}"]),
            float(attrs[f"EN_{loop}_{i}"]),
            int(attrs[f"N_{loop}_{i}"]),
        )
        name = dimension_renamings.get(name, name)
        scan_dimension.append(name)
        scan_shape.append(n)
        scan_coords[name] = np.linspace(start, end, n, endpoint=True)


def find_clean_coords(
    hdu: BinTableHDU,
    attrs: dict[str, Any],
    spectra: Any = None,
    mode: str = "ToF",
    dimension_renamings: dict[str, str] | None = None,
) -> tuple[CoordsDict, dict[str, list[Dimension]], dict[str, Any]]:
    """Determines the scan degrees of freedom, and reads coordinates.

    To do this, we also extract the shape of the actual "spectrum" before reading and parsing
    the coordinates from the header information in the recorded scan.

    Note: because different scan configurations can have different values of the detector
    coordinates, such as for instance when you record in two different angular modes of the
    spectrometer or when you record XPS spectra at different binding energies, we need to be able to
    provide separate coordinates for each of the different scans.

    In the case where there is a unique coordinate, we will return only that coordinate, under the
    anticipated name, such as 'eV'. Otherwise, we will return the coordinates that different between
    the scan configurations under the spectrum name, and with unique names, such as
    'eV-Swept_Spectra0'.

    TODO: Write data loading tests to ensure we don't break MC compatibility

    Args:
        spectra
        hdu
        attrs
        mode: Available modes are "ToF", "MC". This customizes the read process
        dimension_renamings

    Returns:
        A tuple consisting of
        (coordinates, dimensions, np shape of actual spectrum)
    """
    dimension_renamings = (
        dimension_renamings if dimension_renamings else DEFAULT_DIMENSION_RENAMINGS
    )

    scan_coords, scan_dimension, scan_shape = extract_coords(
        attrs,
        dimension_renamings=dimension_renamings,
    )
    logger.debug(f"Found scan shape {scan_shape} and dimensions {scan_dimension}.")

    # bit of a hack to deal with the internal motor used for the swept spectra being considered as
    # a cycle
    if "cycle" in scan_coords and len(scan_coords["cycle"]) > 200:
        logger.debug("Renaming swept scan coordinate to cycle and extracting. This is hack.")
        idx = scan_dimension.index("cycle")

        real_data_for_cycle = hdu.data.columns["null"].array

        scan_coords["cycle"] = real_data_for_cycle
        scan_shape[idx] = len(real_data_for_cycle)

    scan_dimension = [dimension_renamings.get(s, s) for s in scan_dimension[::-1]]
    scan_coords = {dimension_renamings.get(k, k): v for k, v in scan_coords.items()}
    extra_coords = {}
    scan_shape = scan_shape[::-1]

    spectrum_shapes = {}
    dimensions_for_spectra = {}

    spectra = spectra or hdu.columns.names

    if isinstance(spectra, str):
        spectra = [spectra]

    for spectrum_key in spectra:
        logger.debug(f"Considering potential spectrum {spectrum_key}")
        skip_names = {
            lambda name: bool("beamview" in name or "IMAQdx" in name),
        }

        if spectrum_key is None:
            spectrum_key = hdu.columns.names[-1]
            logger.debug(f"Column name was None, using {spectrum_key}")

        if isinstance(spectrum_key, str):
            spectrum_key = hdu.columns.names.index(spectrum_key) + 1

        spectrum_name = hdu.columns.names[spectrum_key - 1]
        loaded_shape_from_header = False
        desc = None

        should_skip = False
        for skipped in skip_names:
            if (callable(skipped) and skipped(spectrum_name)) or skipped == spectrum_name:
                should_skip = True
        if should_skip:
            logger.debug("Skipping column.")
            continue

        try:
            offset = hdu.header[f"TRVAL{spectrum_key}"]
            delta = hdu.header[f"TDELT{spectrum_key}"]
            offset = literal_eval(offset) if isinstance(offset, str) else offset
            delta = literal_eval(delta) if isinstance(delta, str) else delta
            logger.debug(f"Determined (offset, delta): {(offset, delta)}.")

            try:
                shape = hdu.header[f"TDIM{spectrum_key}"]
                shape = literal_eval(shape) if isinstance(shape, str) else shape
                loaded_shape_from_header = True
                logger.debug(f"Successfully loaded coordinate shape from header: {shape}")
            except KeyError:
                shape = hdu.data.field(spectrum_key - 1).shape
                logger.debug(
                    f"Could not use header to determine coordinate shape, using: {shape}",
                )

            try:
                desc = hdu.header[f"TDESC{spectrum_key}"]
                if "(" in desc:
                    # might be a malformed tuple, we can't use literal_eval unfortunately
                    desc = desc.replace("(", "").replace(")", "").split(",")

                if isinstance(desc, str):
                    desc = (desc,)
            except KeyError:
                pass

            if not isinstance(delta, Iterable):
                delta = (delta,)
            if not isinstance(offset, Iterable):
                offset = (offset,)

        except KeyError:
            # if TRVAL{spectrum_key} was not found this means that this column is scalar,
            # i.e. it has only one value at any point in the scan
            spectrum_shapes[spectrum_name] = scan_shape
            dimensions_for_spectra[spectrum_name] = scan_dimension
            continue

        if not scan_shape and shape[0] == 1:
            # the ToF pads with ones on single EDCs
            shape = shape[1:]

        if mode == "ToF":
            rest_shape = shape[len(scan_shape) :]
        elif isinstance(desc, tuple):
            rest_shape = shape[-len(desc) :]
        elif not loaded_shape_from_header:
            rest_shape = shape[1:]
        else:
            rest_shape = shape

        assert len(offset) == len(delta)
        assert len(delta) == len(rest_shape)

        # Build the actually coordinates
        coords = [
            np.linspace(o, o + s * d, s, endpoint=False)
            for o, d, s in zip(offset, delta, rest_shape, strict=True)
        ]

        # We need to do smarter inference here
        def infer_hemisphere_dimensions() -> list[Dimension]:
            # scans can be two dimensional per frame, or a
            # scan can be either E or K integrated, or something I've never seen before
            # try to get the description or the UNIT
            if desc is not None:
                RECOGNIZED_DESCRIPTIONS = {
                    "eV": "eV",
                    "pixels": "pixel",
                    "pixel": "pixel",
                }

                if all(d in RECOGNIZED_DESCRIPTIONS for d in desc):
                    return [RECOGNIZED_DESCRIPTIONS[d] for d in desc]

            try:
                # TODO: read above like desc
                unit = hdu.header[f"TUNIT{spectrum_key}"]
                RECOGNIZED_UNITS = {
                    # it's probably 'arb' which doesn't tell us anything...
                    # because all spectra have arbitrary absolute intensity
                }
                if all(u in RECOGNIZED_UNITS for u in unit):
                    return [RECOGNIZED_UNITS[u] for u in unit]
            except KeyError:
                pass

            # Need to fall back on some human in the loop to improve the read here
            import pdb

            pdb.set_trace()
            return None

        # TODO: for cleanup in future, these should be provided by the implementing endstation
        # class, so they do not get so cluttered, best way will be to make this function a class
        # method, and use class attributes for each of `coord_names_for_spectrum`, etc.
        # For now, patching to avoid error with the microscope camera images at BL7
        coord_names_for_spectrum = {
            "Time_Spectra": ["time"],
            "Energy_Spectra": ["eV"],
            # MC hemisphere image, this can still be k-integrated, E-integrated, etc
            "wave": ["time"],
            "targetPlus": ["time"],
            "targetMinus": ["time"],
            "Energy_Target_Up": ["eV"],
            "Energy_Target_Down": ["eV"],
            "Energy_Up": ["eV"],
            "Energy_Down": ["eV"],
            "Energy_Pol": ["eV"],
        }

        spectra_types = {
            "Fixed_Spectra",
            "Swept_Spectra",
        }

        time_spectra_type = {
            "Time_Target",
        }
        coord_names = None
        if spectrum_name not in coord_names_for_spectrum:
            # Don't remember what the MC ones were, so I will wait to do those again
            # Might have to add new items for new spectrometers as well
            if any(s in spectrum_name for s in spectra_types):
                coord_names = infer_hemisphere_dimensions
            elif any(s in spectrum_name for s in time_spectra_type):
                coord_names = [
                    "time",
                ]
            else:
                import pdb

                pdb.set_trace()
        else:
            coord_names = coord_names_for_spectrum[spectrum_name]

        if callable(coord_names):
            coord_names = coord_names()
            if len(coord_names) > 1 and mode == "MC":
                # for whatever reason, the main chamber records data
                # in nonstandard byte order
                coord_names = coord_names[::-1]
                rest_shape = list(rest_shape)[::-1]
                coords = coords[::-1]

        coords_for_spectrum = dict(zip(coord_names, coords, strict=True))
        # we need to store the coordinates that were kept in a table separately,
        # because they are allowed to differ
        # between different scan configurations in the same file
        if mode == "ToF":
            extra_coords.update(coords_for_spectrum)
        else:
            extra_coords[spectrum_name] = coords_for_spectrum
        dimensions_for_spectra[spectrum_name] = tuple(scan_dimension) + tuple(coord_names)
        spectrum_shapes[spectrum_name] = tuple(scan_shape) + tuple(rest_shape)
        coords_for_spectrum.update(scan_coords)

    extra_coords.update(scan_coords)

    if mode != "ToF":
        detector_coord_names = [k for k, v in extra_coords.items() if isinstance(v, dict)]

        from collections import Counter

        c = Counter(item for name in detector_coord_names for item in extra_coords[name])
        conflicted = [k for k, v in c.items() if v != 1 and k != "cycle"]

        flat_coordinates = collect_leaves(extra_coords)

        def can_resolve_conflict(c):
            coordinates = flat_coordinates[c]

            if not isinstance(coordinates, list) or len(coordinates) < 2:
                return True

            # check if list of arrays is all equal
            return functools.reduce(
                lambda x, y: (np.array_equal(x[1], y) and x[0], y),
                coordinates,
                (True, coordinates[0]),
            )[0]

        conflicted = [c for c in conflicted if not can_resolve_conflict(c)]

        def clarify_dimensions(dims: list[Dimension], sname: str) -> list[Dimension]:
            return [d if d not in conflicted else d + "-" + sname for d in dims]

        def clarify_coordinate(
            coordinates: CoordsDict | ndarray,
            sname: str,
        ) -> CoordsDict | ndarray:
            if not isinstance(coordinates, dict):
                return coordinates

            return {
                k if k not in conflicted else k + "-" + sname: v for k, v in coordinates.items()
            }

        dimensions_for_spectra = {
            k: clarify_dimensions(v, k) for k, v in dimensions_for_spectra.items()
        }
        extra_coords = {k: clarify_coordinate(v, k) for k, v in extra_coords.items()}
        extra_coords = dict(iter_leaves(extra_coords))

    return extra_coords, dimensions_for_spectra, spectrum_shapes
