Major Changes from 3.0.1

- A new attrs, "energy_notation" is introduced, which determine the "Kinetic" energy or "Binding" energy
- Use correct way to convert from the angle to momentum. (Original way is incorrect)
- filters.boxcar: skip_nan is removed
- Dataset.S.spectra returns the list of the xr.DataArrays whose dims contains "eV". (See xarray_extensions.py)

Fix from 3.0.1

- bug of concatenating in broadcast_model
- Import error in BZ plotting example #7

Minor Changes from 3.0.1

- Remove beta arg from filters.curvature
