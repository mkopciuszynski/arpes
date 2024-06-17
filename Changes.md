# Major Changes from 3.0.1

- Most important change:

  - Use correct method to convert from the angle to momentum. (The original way
    was incorrect. And I find most of all python related libraries in github takes
    the wrong algorithm for converting. I don't want any more incorrect knowledge
    to spread, but there's nothing I can do about it. Sigh.)

- New feature

  - Provide SPD_main.py & prodigy_itx.py
  - Provide IF\_\_UMCS.py & prodigy_xy.py
  - Introduce Type annotation, which is major trend in the current python coding.
    - Users are requested to know type of the object treated. Especially,
      the users should know the difference between xr.DataArray & xr.Dataset
  - Introduce a new attrs, "energy_notation", which determines either "Kinetic"
    energy or "Binding" energy
    - Add new method: S.switch_energy_notation(self, nonlinear_order=1)
  - More comatible with ASE.

    - > = v3.23.0 is required.

  - Dataset.S.spectra returns the list of the xr.DataArrays whose dims contains
    "eV". (See xarray_extensions.py)
  - Add a new method S.swap_angle_unit() to change the angle unit (deg <-> radian)
    For presenting the data, the degrees unit is more familiar.
  - Replace algorithms to make them simpler and more efficient

    - stack_plot.py/flat_stack_plot
    - analysis/general.py/rebin

- Coding guide line

  - Do not carelessly set default=None
  - Not pursuing the graphinca user interface.
    - In most case, just want to know the value of the coordinate when GUI is
      needed, which is enough for using current GUI interface including hvplot.
      And if GUI is essentially required (for not pythonic user?), using igor
      would be best. (You don't have to go that far to use Python.)

- Removing

  Many files, methods, and classes have been removed due to inconsistency, legacy
  style or been deemed useless. I believe that most of these were not use in
  version 3. Here is just a short list of the ones I remember now:

  - Remove arpes.all

    - Certainly, this it is indeed a lazy and carefree approach, but it's too
      rough method that leads to a bugs and does not match the current pythonic style.

  - Remove utilities/attrs.py

    - The functions in this module have not been used and are unlikely to be
      used in the future.

  - Remove fits/fit_model/peaks.py

    - The classes (fitting models) defined in this module are essentially
      needless, **as you can use the + operator on the Model instances.**

  - modules that use the Bokeh.

    There is a dependency problem among bokeh, tornard, and Jupyter, which I
    cannot fix because I'm haven't use Bokeh. But note that hvplot can work with
    the current version.

    - arpes/plotting/band_tool.py
    - arpes/plotting/curvature_tool.py
    - arpes/plotting/fit_inspection_tool.py
    - arpes/plotting/comparison_tool.py
    - arpes/plotting/dyn_tool.py
    - arpes/plotting/interactive_utils.py
    - arpes/plotting/interactive.py
    - arpes/plotting/path_tool.py
    - arpes/plotting/mask_tool.py

  - Remove MappableDict class
  - Remove overlapped_stack_dispersion_plot
    - use stack_dispersion_plot with appropriate args
  - Remove G.extent in xarray_extensions, which is not so usuful
  - Remove scan_row property
  - Remove original_id method, as I cannot figure out the purpose.
  - Remove lmfit_plot.py. The original ModelResult.plot() is sufficiently
    useful, and no reason for keeping to maintain this patched version.
  - Remove condensed_attrs: We should not enfoce camelcase on attributes,
    while original version did. Rather, the snake_case would be better from the
    modern pythonic viewpoint.
  - Remove `trace` arg for debugging. This technique may be sharp, but not so
    well fitted the current python trend; typing oriented.

  - Remove the class and functions in corrections/**init**.py (HashableDict,
    reference_key, correction_from_reference_set), which have not used.
  - Remove shift_gamma from slice_along_path
  - Furthermore, added Deprecated tag to the functions that will be removed in
    the future.
  - Remove syntax-sugar-like methods in xarray_extensions.

    - S.fermi_surface, S.generic_fermi_surface
    - S.along
    - G.iterate_axis ( which is just a combination of .G.iter_coords & .sel methods)

  - Remove most of methods in xr.Dataset.F.

    - Most methods in Dataset.F.method are just a wrapper of
      Dataset.resutls.F.method. Considering the difference between xr.Dataset
      and xr.DataArray, such xr.Dataset.F.method is not meaningful.

  - Remove polarization_plot, which is just a wrapper of spin_polarized_spectrum()

Fix from 3.0.1

- bug of concatenating in broadcast_model
- Import error in BZ plotting example #7

Minor Changes from 3.0.1

- Remove beta arg from filters.curvature
