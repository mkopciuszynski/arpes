"""Implements data loading for the Lanzara group Spin-ToF."""

from __future__ import annotations

# pylint: disable=no-member
import itertools
import warnings
from logging import DEBUG, INFO
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
import xarray as xr
from astropy.io import fits

from arpes.debug import setup_logger
from arpes.endstations import EndstationBase, ScanDesc, find_clean_coords
from arpes.provenance import Provenance, provenance_from_file
from arpes.utilities import rename_keys
from arpes._typing import Spectrometer
from arpes import DATA_PATH

__all__ = ("SpinToFEndstation",)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


class SpinToFEndstation(EndstationBase):
    """Implements data loading for the Lanzara group Spin-ToF."""

    PRINCIPAL_NAME = "ALG-SToF"
    ALIASES: ClassVar[list[str]] = ["ALG-SToF", "SToF", "Spin-ToF", "ALG-SpinToF"]
    SKIP_ATTR_FRAGMENTS: ClassVar[set[str]] = {
        "MMX",
        "TRVAL",
        "TRDELT",
        "COMMENT",
        "OFFSET",
        "SMOTOR",
        "TUNIT",
        "PMOTOR",
        "TDESC",
        "NAXIS",
        "TTYPE",
        "TFORM",
        "XTENSION",
        "BITPIX",
        "TDELT",
        "TRPIX",
    }

    COLUMN_RENAMINGS: ClassVar[dict[str, str]] = {
        "TempA": "temperature_cryo",
        "TempB": "temperature_sample",
        "Current": "photocurrent",
        "ALS_Beam_mA": "beam_current",
        "Energy_Spectra": "spectrum",
        "targetPlus": "t_up",
        "targetMinus": "t_down",
        "wave": "spectrum",  # this should not occur simultaneously with 'Energy_Spectra'
        "Time_Target_Up": "t_up",
        "Time_Target_Down": "t_down",
        "Energy_Target_Up": "up",
        "Energy_Target_Down": "down",
        "Photocurrent_Up": "photocurrent_up",
        "Photocurrent_Down": "photocurrent_down",
        "Phi": "phi",
    }

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "LMOTOR0": "x",
        "LMOTOR1": "y",
        "LMOTOR2": "z",
        "LMOTOR3": "theta",
        "LMOTOR4": "beta",
        "LMOTOR5": "chi",
        "LMOTOR6": "delay",
        "Phi": "phi",
    }

    MERGE_ATTRS: ClassVar[Spectrometer] = {
        "length": 1.1456,  # This isn't correct but it should be a reasonable guess
        "mstar": 0.5,
    }

    def load_SToF_hdf5(self, scan_desc: ScanDesc | None = None) -> xr.Dataset:
        """Imports a FITS file that contains ToF spectra.

        Args:
            scan_desc: Dictionary with extra information to attach to the xr.Dataset, must contain
            the location of the file
            kwargs: Not used in the current version.

        Returns:
            The loaded data.
        """
        if scan_desc is None:
            scan_desc = {}
        data_loc = Path(scan_desc.get("path", scan_desc.get("file", "")))
        if not data_loc.is_absolute():
            assert DATA_PATH is not None
            data_loc = Path(DATA_PATH) / data_loc

        f = h5py.File(data_loc, "r")

        dataset_contents = {}
        raw_data = f["/PRIMARY/DATA"][:]
        raw_data = raw_data[:, ::-1]  # Reverse the timing axis
        dataset_contents["raw"] = xr.DataArray(
            raw_data,
            coords={"x_pixels": np.linspace(0, 511, 512), "t_pixels": np.linspace(0, 511, 512)},
            dims=("x_pixels", "t_pixels"),
            attrs=f["/PRIMARY"].attrs.items(),
        )
        pronance_context: Provenance = {
            "what": "Loaded Anton and Ping DLD dataset from HDF5.",
            "by": "load_DLD",
        }

        provenance_from_file(dataset_contents["raw"], str(data_loc), pronance_context)

        return xr.Dataset(dataset_contents, attrs=scan_desc)

    def load_SToF_fits(self, scan_desc: ScanDesc) -> xr.Dataset:
        """Loads FITS convention SToF data.

        The data acquisition software is rather old, so this has to handle data formats
        from early versions of the E. Rotenberg software. Some similarities exist with the
        main chamber loading code.

        Args:
            scan_desc: [TODO:description]
        """
        data_loc = Path(scan_desc.get("path", scan_desc.get("file", "")))
        if not data_loc.exists():
            assert DATA_PATH is not None
            data_loc = Path(DATA_PATH) / data_loc

        hdulist = fits.open(data_loc)

        hdulist[0].verify("fix+warn")
        header_hdu, hdu = hdulist[0], hdulist[1]

        scan_desc.update(dict(hdu.header))
        scan_desc.update(dict(header_hdu.header))

        drop_attrs = ["COMMENT", "HISTORY", "EXTEND", "SIMPLE", "SCANPAR", "SFKE_0"]
        for dropped_attr in drop_attrs:
            if dropped_attr in scan_desc:
                del scan_desc[dropped_attr]

        coords, dimensions, spectrum_shape = find_clean_coords(hdu, scan_desc)
        dimensions = {
            k: [SpinToFEndstation.RENAME_KEYS.get(n, n) for n in v] for k, v in dimensions.items()
        }
        coords = rename_keys(coords, SpinToFEndstation.RENAME_KEYS)

        columns = hdu.columns

        spin_column_names = {
            "targetMinus",
            "targetPlus",
            "Time_Target_Up",
            "Time_Target_Down",
            "Energy_Target_Up",
            "Energy_Target_Down",
            "Photocurrent_Up",
            "Photocurrent_Down",
        }

        is_spin_resolved = any(cname in columns.names for cname in spin_column_names)
        spin_columns = ["CurrentTempA", "TempB", "ALS_Beam_mA", *list(spin_column_names)]
        straight_columns = ["Current", "TempA", "TempB", "ALS_Beam_mA", "Energy_Spectra", "wave"]
        take_columns = spin_columns if is_spin_resolved else straight_columns

        # We could do our own spectrum conversion too, but that would be more annoying
        # it would slightly improve accuracy though
        spectra_names = [name for name in take_columns if name in columns.names]

        skip_predicates = {
            lambda k: any(s in k for s in self.SKIP_ATTR_FRAGMENTS),
        }

        scan_desc = {
            k: v for k, v in scan_desc.items() if not any(pred(k) for pred in skip_predicates)
        }
        scan_desc = rename_keys(scan_desc, SpinToFEndstation.RENAME_KEYS)

        # TODO: we should try to unify this with the FITS file loader,
        # but there are a few current inconsistencies
        data_vars = {}

        for spectrum_name in spectra_names:
            column_shape = spectrum_shape[spectrum_name]
            data_for_resize = hdu.data.columns[spectrum_name].array

            try:
                # best possible case is that we have identically all of the data
                resized_data = data_for_resize.reshape(column_shape)
            except ValueError:
                # if we stop scans early, the header is already written and so the size of the data
                # will be small along the experimental axes
                rest_column_shape = column_shape[1:]
                n_per_slice = int(np.prod(rest_column_shape))
                total_shape = data_for_resize.shape
                total_n = np.prod(total_shape)

                n_slices = total_n // n_per_slice

                if total_n // n_per_slice != total_n / n_per_slice:
                    # the last slice was in the middle of writing when something hit the fan
                    # we need to infer how much of the data to read, and then repeat the above
                    # we need to cut the data

                    # This can happen when the labview crashes during data collection,
                    # we use column_shape[1] because of the row order that is used in the FITS file
                    data_for_resize = data_for_resize[
                        0 : (total_n // n_per_slice) * column_shape[1]
                    ]
                    warnigs_msg = "Column {} was in the middle of slice when DAQ stopped. "
                    warnigs_msg += "Throwing out incomplete slice..."
                    warnings.warn(
                        warnigs_msg.format(
                            spectrum_name,
                        ),
                        stacklevel=2,
                    )

                column_shape = list(column_shape)
                column_shape[0] = n_slices

                try:
                    resized_data = data_for_resize.reshape(column_shape)
                except Exception:
                    # we should probably zero pad in the case where the slices are not the right
                    # size
                    logger.exception("Exception Occur")
                    continue

                altered_dimension = dimensions[spectrum_name][0]
                coords[altered_dimension] = coords[altered_dimension][:n_slices]

            data_vars[spectrum_name] = (
                dimensions[spectrum_name],
                resized_data,
                scan_desc,
            )

        data_vars = rename_keys(data_vars, SpinToFEndstation.COLUMN_RENAMINGS)
        if "beam_current" in data_vars and np.all(data_vars["beam_current"][1] == 0):
            # Wasn't taken at a beamline
            del data_vars["beam_current"]

        hdulist.close()

        relevant_dimensions = {
            k for k in coords if k in set(itertools.chain(*[_[0] for _ in data_vars.values()]))
        }
        relevant_coords = {k: v for k, v in coords.items() if k in relevant_dimensions}

        dataset = xr.Dataset(
            data_vars,
            relevant_coords,
            scan_desc,
        )

        for data_arr in dataset.data_vars.values():
            if "time" in data_arr.dims:
                data_arr.data = data_arr.sel(time=slice(None, None, -1)).data
        provenance_context: Provenance = {
            "what": "Loaded Spin-ToF dataset",
            "by": "load_DLD",
        }

        provenance_from_file(dataset, str(data_loc), provenance_context)

        return dataset

    def load(self, scan_desc: ScanDesc) -> xr.Dataset:
        """Loads Lanzara group Spin-ToF data.

        Args:
            scan_desc: [TODO:description]

        Raises:
            TypeError: [TODO:description]
        """
        if scan_desc is None:
            warnings.warn(
                "Attempting to make due without user associated scan_desc for the file",
                stacklevel=2,
            )
            msg = "Expected a dictionary of scan_desc with the location of the file"
            raise TypeError(msg)
        data_loc = scan_desc.get("path", scan_desc.get("file"))
        scan_desc = {
            k: v for k, v in scan_desc.items() if not isinstance(v, float) or not np.isnan(v)
        }

        if Path(data_loc).suffix == ".fits":
            return self.load_SToF_fits(scan_desc)

        return self.load_SToF_hdf5(scan_desc)
