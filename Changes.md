Major Changes from 3.0.1

- Introdcue a new attrs, "energy_notation" is introduced, which determines eithe "Kinetic" energy or "Binding" energy
  - Add new method: S.switch_energy_notation(self, nonlinear_order=1)
- Use correct method to convert from the angle to mthe omentum. (Original approach was found to be incorrect)
- filters.boxcar: skip_nan has been removed
- Dataset.S.spectra returns the list of the xr.DataArrays whose dims contains "eV". (See xarray_extensions.py)
- Provide SPD_main.py & prodigy_itx.py
- Add a new method S.swap_angle_unit() to change the angle unit (deg <-> radian)

- Replace algorithms to make them simpler and more efficient
  - stack_plot.py/flat_stack_plot
  - analysis/general.py/rebin

Fix from 3.0.1

- bug of concatenating in broadcast_model
- Import error in BZ plotting example #7

Minor Changes from 3.0.1

- Remove beta arg from filters.curvature

