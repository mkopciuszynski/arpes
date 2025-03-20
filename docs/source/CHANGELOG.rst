Changelog
=========

Changes are listed with most recent versions at the top.

Dates are in YYYY/MM/DD format.

Primary (X.-.-) version numbers are used to denote backwards
incompatibilities between versions, while minor (-.X.-) numbers
primarily indicate new features and documentation.

4.2.4 (XXXX-XX-XX)
^^^^^^^^^^^^^^^^^^

Changed
~~~~~~~

* Rename inset_cut_locator -> insert_cut_locator
* Remove sum_annotation and mean_annotation in plotting.utils
* Remove to_arrays in xarray_extensions.py because of too-simple.
* Remove src/arpes/optics.py
  - This should not be included in pyarpes, because it is not closely related to arpes analysis.
* Remove src/arpes/utilities/image.py
  - In certain special situations, this may be a meaningful function. However, as a practical matter, direct reading of image data is not likely to produce meaningful data (data that can be used in a paper).
* Remove arpes.exceptions: pragmatically, it has not been used. And I don't think the own class for exception is good idea.
* Remove arg keep_parent_ref in provenance and provenance_multiple_parents
* Remove load_data_for_figure from plotting.utils
* Remove CoincidentLinePlot (Same (at least similar) feature can be done with fill_between)
* Remove zero_nans arg in shift_by. Use da.fillna(0), instead.
* Remove cut_nan_coords. Use dropna like that:
  for cname in da.coords:
    da = da.dropna(dim=cname, how="any")

* Move utilities/convert/trapezoid.py -> correction/trapezoid.py

* Change method name: S.swap_angle_unit -> S.switch_angle_unit

* Deprecated
  - S.correct_angle_by
  - S.corrected_angle_by
  - S.transpose_to_front
  - S.transpose_to_back
  - S.to_arrays
  - S.scan_degrees_of_freedom, S.degrees_of_freedom and spectrum_degrees_of_freedom
  - arrange_by_indices and unarrange_by_indices (defined in utilities/__init__.py). They are not used internally.

* Refactoring

  * ConvertTrapezoidalCorrection: More general

4.2.3 (2025-1-5)
^^^^^^^^^^^^^^^^^^

Changed
~~~~~~~

* Deprecated: arrange_by_indices, unarrange_by_indice in utilities/__init__ 

4.2.2 (2024-12-31)
^^^^^^^^^^^^^^^^^^

Changed
~~~~~~~

* Update S.fat_sel.
* Add figure_gallery.ipynb
* Rename IF_UMCS to DSN_UMCS

4.2.1 (2024-12-26)
^^^^^^^^^^^^^^^^^^

Changed
~~~~~~~

* Changed internally.  repair.py is removed.  API does not change.

4.2.0 (2024-12-25)
^^^^^^^^^^^^^^^^^^

Changed
~~~~~~~

* Not recommend to use the number to identify the file in io.load_data.
* Update plot_movie(), build_crosscorrelation and add a new function, plot_movie_with_appropriate_args() (#63, #67, #68)

* Remove keep_parent_ref arg 

* Method name change
  * S.correct_angle_by -> S.correct_coords [DeprecatedWarning]
  * S.corrected_angle_by -> S.corrected_coords [DeprecatedWarning]
  * S.transpose_to_front -> S.transpose_to_front [DeprecatedWarning]

* Deprecated method 
  * transpose_to_front  (Use standard Xarray transform)  [DeprecatedWarning] (#65)
  * transpose_to_back (Use standard Xarray transform)    [DeprecatedWarning] (#65)

* Remove unused or not so deeply related to ARPES file
  * optics.py, exceptions.py, and images.py

  
Minor
~~~~~

* Method name change
  These methods are used internally, so most users don't need to care the change.

  * S.shift_coords -> shift_meshgrid
  * S.scale_coords -> scale_meshgrid
  * S.transform_coords -> transform_meshgrid


* Chenge the energy notation name (Kintic -> Final)
  This is used internally, so most users don't need to care the change.

4.1.0 (2024-11-20)
^^^^^^^^^^^^^^^^^^

Changed
~~~~~~~

* As this version, we can begin with `import arpes`.
* Add new endstation
* Logging and endstation plugin can be selected from local_config.py

Minor
~~~~~

* Recommend to use uv, instead of rye.

4.0.1 (2024-07-21)
^^^^^^^^^^^^^^^^^^

Changed
~~~~~~~

The name change

* corrections -> correction

     This is somehow backwards incopatibilities.  However, the effect for most is really minor,
     because this functionalities are not so frequently used. Thus the major version number has not been changed.

* New UI

  * concat_along_phi_ui (based on holoviews)
 
* Remove Qt related modules.  (Move them to legacy_files)


4.0.0 (2024-07-12)
^^^^^^^^^^^^^^^^^^

New
~~~

Non-original author update.

* Provide SPD_main.py and prodigy_itx.py to load the data measured with SPECS prodigy.
* A new method S.swap_angle_unit() to change the angle unit (deg <-> radian)

Changed
~~~~~~~
* Required python version >= 3.11
* Introduce type hints.  
    - Type annotation sometimes limits the python's flexibility, but for induces the robustness, which is essentially useful for scientific analyais.
    - Because of the same reason, drop the "flexibile" (from a certain view points, it may be pythonic) codes.
* Drop PyQt5, use PySide6 instead.
* Drop the Bokeh based functionalities because they are not compatible with the current jupyter. 
    - curvature_tool, comparison_tool
    - While some of them might be useful, but at least when I started to use this, the compatibility was broken.  I don't know how these were useful.
* Remove arpes.all
* And very many breaking Changes.
    * Remove G.extent
    * Remove overlapped_stack_dispersion_plot
        - use stack_dispersion_plot_with_appropriate_args
    * Revise the k-conversion.  The original version is correct from the view of the coding, but incorrect from the physics!
    * introduce new attrs, "energy_notation". if not specified, attrs["energy_notation"] = "Binding" is assumed to keep the consistency from the previous version.

    * see Changes.md for others

In coding style:

* Drop carelessly set default=None

Fixed
~~~~~
* broadcast_model concatenation error #18  (https://github.com/chstan/arpes/issues/18)
* Fix error in BZ plot due to the recent version of ASE #20
* Fix the import error in BZ plotting example #7



3.0.1 (2021-07-27)
^^^^^^^^^^^^^^^^^^

New
~~~

Changed
~~~~~~~

Added tests for momentum conversion and for Qt tools.

Fixed
~~~~~

Bugfix release to fix Qt API after bumping Qt versions.
Tests have been added which hit large parts of the Qt code
to prevent problems like this in the future.

3.0.0 (2021-07-27)
^^^^^^^^^^^^^^^^^^^

New
~~~

1. Numba has been adopted to accelerate k-space conversion resulting in 
   10-50x speedup compared to the older plain numpy versions of code.
2. Additional example data has been added so that representative ARPES data
   covering standard types of experiments are available.
3. The documentation site has been moved from Netlify to https://arpes.readthedocs.io/
   and the content available greatly expanded.

   * Tutorials for common types of analysis are available as Jupyter notebooks.
   * An organized API documentation page is available.
   * Docstrings have been massively expanded to cover the public API
     and most of the internal API.
   * The documentation build process has been simplified.

4. The momentum conversion API has been expanded with utility functions
   
   * ``arpes.utilities.conversion.forward.convert_through_angular_point``: Performs
     a cut in momentum at a particular angle and passing through the angular coordinate 
     provided.
   * ``arpes.utilities.conversion.forward.convert_through_angular_pair``: Performs 
     a cut in momentum passing through two given angular coordinates.

   These are very helpful in getting high symmetry cuts rapidly.

5. Deep learning utilities upstreamed.
6. Multithreaded curve fitting.
7. Fit introspection utilities upstreamed.
8. Numerous small but compatible changes to the public API.

Changed
~~~~~~~

1. The xarray data accessor previously at .T has been named to .G to
   prevent shadowing the transpose function.
2. pylint -> black
3. Bump dependency versions, largely due to compatibility requirements
   with pyqtgraph.
4. Old .csv/spreadsheet driven APIs removed.

Fixed
~~~~~

1. Circular references have been removed from tools which use Qt which
   previously lead to crashes due to objects being freed in C++/Qt5 but
   retained in Python/PyQt5.

   Additionally, some diagnostics have been added to help deal with
   similar problems in the future.

.. _section-1:

2.6.0 (2020-1-20)
^^^^^^^^^^^^^^^^^

.. _new-1:

New
~~~

1. Igor loader, aliased to ‘pxt’, ‘wave’, etc.

.. _changed-1:

Changed
~~~~~~~

1. Improved documentation and intro videos

.. _fixed-1:

Fixed
~~~~~

1. Made loading pxt files more stable by adding a utility to safely
   decode strings when the encoding used is not known but is a common
   format

.. _section-2:

2.5.0 (2019-12-5)
^^^^^^^^^^^^^^^^^

.. _new-2:

New
~~~

1. Added a Qt-based waypoint data browser similar to what’s available at
   the Spectromicroscopy beamline, ``path_tool``.
2. Added a Qt-based masking tool ``mask_tool``
3. Added a Qt-based background subtraction tool ``bkg_tool``.
4. Generic Qt tools that interact with “paths” or “regions” are now
   simple to add with ``CoreTool``

.. _changed-2:

Changed
~~~~~~~

1. Unitful axes on all Qt-based utilities

.. _section-3:

2.4.0 (2019-11-24)
^^^^^^^^^^^^^^^^^^^

.. _new-3:

New
~~~

1. Data loading code for the Spectromicroscopy beamline at Elettra.
2. Added a number of interactive utilities
3. Documentation/tutorial on adding interactive utilities
4. ``qt_ktool``
5. Borrow code from DAQuiri for UI generation

.. _changed-3:

Changed
-------

1. Improved the documentation and FAQ.
2. Refactor file finding to support subfolders and endstation specific
   behavior

.. _section-4:

2.3.0 (2019-10-28)
^^^^^^^^^^^^^^^^^^^^

.. _new-4:

New
~~~

1. More moiré analysis tools including commensurability measures.
2. ``FallbackEndstation``, see the changed section below.

.. _changed-4:

Changed
-------

Serious refactor to data loading. On the surface not much is different,
except that most things are more permissive by default now. In
particular, you can often get away with not passing the ``location=``
keyword but it is recommended still.

There is now a ``FallbackEndstation`` that tries to determine which
endstation to use in the case of missing ``location`` key. This is to
reduce the barrier to entry for new users.

.. _fixed-2:

Fixed
-----

.. _section-5:

2.2.0 (2019-08-21)
^^^^^^^^^^^^^^^^^^^^

.. _new-5:

New
~~~

1. Moiré analysis module with some code to generate primitive moiré unit
   cells and plot them
2. Subpixel alignment in 1D and 2D based on image convolution and
   quadratic fitting this is useful for tracking and correcting shifts
   in valence data due to work function changes, charging, etc.
3. More or less fully fledged k-independent self energy analysis module
   (arpes.analysis.self_energy)
4. BZ exploration tool
5. Large refactor to data provenance

   1. Now guaranteed produced for every plot using ``savefig``
   2. By default we configure IPython to log all code execution
   3. Most recent cell/notebook evaluations are included in provenance
      information

6. ``convert_coordinates`` is now nearly an inverse transform to
   ``convert_to_kspace`` on the coordinates as is appropriate. In
   particular, this conversion is exact as opposed to small angle
   approximated

Minor
~~~~~

1. Some wrappers around getting Jupyter/IPython state
2. ``imread`` wrapper that chooses backend between ``imageio`` and
   ``cv2``
3. Plotting utilities

   1. ``dark_background`` context manager changes text and spines to
      white
   2. Data unit/axis unit conversions (``data_to_axis_units`` and
      friends)
   3. ``mean_annotation`` as supplement to ``sum_annotation``

4. ``xarray_extensions``:

   1. ``with_values`` -> generates a copy with replaced data
   2. ``with_stanard_coords`` -> renames deduped (``eV-spectrum0`` for
      instance) coords back to standard on a xr.DataArray
   3. ``.logical_offsets`` calculates logical offsets for the ‘x,y,z’
      motor set
   4. Correctly prefers ``hv`` from coords now
   5. ``mean_other`` as complement to ``sum_other``
   6. ``transform``: One ``map`` to rule them all

.. _changed-5:

Changed
~~~~~~~

.. _fixed-3:

Fixed
~~~~~

.. _section-6:

2.1.4 (2019-08-07)
^^^^^^^^^^^^^^^^^^^^^^

.. _new-6:

New
~~~

.. _changed-6:

Changed
~~~~~~~

1. Prevent PyPI builds unless conda build succeeds, so that we can have
   a single package-time test harness (run_tests.py).

.. _fixed-4:

Fixed
~~~~~

1. Fix documentation to better explain conda installation. In
   particular, current instructions avoid a possible error arising from
   installing BLAS through conda-forge.

2. colorama now listed as a dependency in conda appropriately.

.. _section-7:

2.1.3 (2019-08-07)
^^^^^^^^^^^^^^^^^^^

.. _new-7:

New
~~~

.. _changed-7:

Changed
~~~~~~~

1. ``pylint``\ ed

.. _fixed-5:

Fixed
~~~~~

1. Fix manifest typo that prevents example data being included

.. _section-8:

2.1.2 (2019-08-06)
^^^^^^^^^^^^^^^^^^^^

.. _new-8:

New
~~~

.. _changed-8:

Changed
~~~~~~~

.. _fixed-6:

Fixed
~~~~~

1. Removed type annotation for optional library breaking builds

.. _section-9:

2.1.1 (2019-08-06)
^^^^^^^^^^^^^^^^^^^^^

.. _new-9:

New
~~~

1. Improved type annotations
2. Slightly safer data loading in light of plugins: no need to call
   ``load_plugins()`` manually.

.. _changed-9:

Changed
~~~~~~~

.. _fixed-7:

Fixed
~~~~~

1. Data moved to a location where it is available in PyPI builds

.. _section-10:

2.1.0 (2019-08-06)
^^^^^^^^^^^^^^^^^^^^^

.. _new-10:

New:
~~~~

1. Improved API documentation.
2. Most recent interactive plot context is saved to
   ``arpes.config.CONFIG['CURRENT_CONTEXT']``. This allows simple and
   transparent recovery in case you forget to save the context and
   performed a lot of work in an interactive session. Additionally, this
   means that matplotlib interactive tools should work transparently, as
   the relevant widgets are guaranteed to be kept in memory.
3. Improved provenance coverage for builtins.

.. _changed-10:

Changed:
~~~~~~~~

1. Metadata reworked to a common format across all endstations. This is
   now documented appropriately with the data model.

.. _fixed-8:

Fixed:
~~~~~~

1. MBS data loader now warns about unsatisfiable attributes and produces
   otherwise correct coordinates in the PyARPES format.
2. Some improvements made in the ANTARES data loader, still not as high
   quality as I would like though.

.. _section-11:

2.0.0 (2019-07-31)
^^^^^^^^^^^^^^^^^^^^^^

.. _new-11:

New:
~~~~

1. Major rework in order to provide a consistent angle convention

2. New momentum space conversion widget allows setting offsets
   interactively

3. Fermi surface conversion functions now allow azimuthal rotations

4. New ``experiment`` module contains primitives for exporting scan
   sequences. This is an early addition towards being able to perform
   ARPES experiments from inside PyARPES.

   1. As an example: After conducting nano-XPS, you can use PCA to
      select your sample region and export a scan sequence just over the
      sample ROI or over the border between your sample and another
      area.

.. _changed-11:

Changed:
~~~~~~~~

1. All loaded data comes with all angles and positions as coordinates
2. All loaded data should immediately convert to momentum space without
   issue (though normal emission is not guaranteed!)
3. Documentation changes to reflect these adjustments to the data model

.. _fixed-9:

Fixed:
~~~~~~

1. Documentation link in README.rst is now correct.

.. _section-12:

1.2.0 (2019-07-18)
^^^^^^^^^^^^^^^^^^^^^

.. _new-12:

New:
~~~~

1. Ship example data so that people can try what is in the documentation
   immediately after installing
2. Users can now load data directly, i.e. without a spreadsheet, with
   ``load_without_dataset``, in the future this will support matches
   based on the current working directory.
3. Users are better warned when spreadsheets are not in the correct
   format. Spreadsheet loading is also generally more permissive, see
   below.

.. _changed-12:

Changed:
~~~~~~~~

1. Added more tests, especially around data loading, spreadsheet loading
   and normalization.

.. _fixed-10:

Fixed:
~~~~~~

1. Spreadsheet loading no longer relatively silently fails due to
   whitespace in column names, we might nevertheless consider doing more
   significant cleaning of data at the very initial stages of
   spreadsheet loading.
2. Spreadsheet loading now appropriately uses safe_read universally.
   ``modern_clean_xlsx_dataset`` is functionally deprecated, but will
   stay in at least for a little while I consider its removal.
3. Spreadsheet loading now appropriately handles files with ‘cleaned’ in
   their name.
4. Spreadsheet writing will not include the index and therefore an
   unnamed column when saving to disk.

.. _section-13:

1.1.0 (2019-07-11)
^^^^^^^^^^^^^^^^^^^

.. _new-13:

New:
~~~~

1. Add a self-check utility for debugging installs,
   ``import arpes; arpes.check()``
2. PyARPES can generate scan directives to make working at beamlines or
   nanoARPES endstations simpler. You can now export a region or
   boundary of a region from a PyARPES analysis to a (first pass)
   LabView compatible scan specification. For now this consists of a
   coordinate list and optional spectrum declaration.
3. ``local_config.py`` now has a programmatic interface in
   ``arpes.config.override_settings``.
4. Add ``arpes.utilities.collections.deep_update``

.. _changed-13:

Changed:
~~~~~~~~

1. Documentation overhaul, focusing on legibility for new users and
   installation instructions

.. _fixed-11:

Fixed:
~~~~~~

1. Version requirements on ``lmfit`` are now correct after Nick added
   ``SplitLorentzian`` xarray compatible models

.. _section-14:

1.0.2 (2019-07-08)
^^^^^^^^^^^^^^^^^^^

.. _new-14:

New:
~~~~

1. Moved to CI/CD on Azure Pipelines
   (https://dev.azure.com/lanzara-group/PyARPES)
2. Tests available for data loading and some limited analysis routines

.. _changed-14:

Changed:
~~~~~~~~

1. Lanzara group Main Chamber data loading code will set a photon energy
   of 5.93 eV on all datasets by default

.. _fixed-12:

Fixed:
~~~~~~

1. ``arpes.analysis.derivative.dn_along_axis`` now properly accepts a
   smoothing function (``smooth_fn``) with the signature
   ``xr.DataArray -> xr.DataArray``.

1.0.0 (June 2019)
^^^^^^^^^^^^^^^^^

.. _new-15:

New:
~~~~

1. First official release. API should be largely in place around most of
   PyARPES.
