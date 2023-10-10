"""Standard plotting routines and utility code for ARPES analyses."""
from __future__ import annotations  # noqa: I001

from .stack_plot import stack_dispersion_plot, flat_stack_plot, offset_scatter_plot
from .movie import plot_movie

"""
from .annotations import annotate_cuts, annotate_experimental_conditions, annotate_point
from .bands import plot_with_bands
from .basic import make_reference_plots
from .dispersion import (
    cut_dispersion_plot,
    fancy_dispersion,
    hv_reference_scan,
    labeled_fermi_surface,
    plot_dispersion,
    reference_scan_fermi_surface,
    scan_var_reference_plot,
)
from .dos import plot_core_levels, plot_dos
from .fermi_edge import fermi_edge, plot_fit
from .fermi_surface import fermi_surface_slices, magnify_circular_regions_plot
from .mask_tool import mask
from .parameter import plot_parameter
from .spatial import plot_spatial_reference, reference_scan_spatial
from .spin import spin_polarized_spectrum, spin_colored_spectrum, spin_difference_spectrum
from .utils import (
    savefig,
    remove_colorbars,
    fancy_labels,
)
"""
"""
# Qt based
from .band_tool import BandTool
from .qt_tool import qt_tool
from .qt_ktool import ktool
"""

"""
# Bokeh based
from .curvature_tool import CurvatureTool
from .dyn_tool import DynamicTool, dyn
from .fit_inspection_tool import FitCheckTool
from .interactive import ImageTool
from .path_tool import path_tool
from .comparison_tool import compare
"""
