"""Standard plotting routines and utility code for ARPES analyses."""

# pyright: reportUnusedImport=false
from __future__ import annotations

from .annotations import annotate_cuts, annotate_experimental_conditions, annotate_point
from .bands import plot_with_bands
from .basic import make_overview, make_reference_plots
from .decoration import h_gradient_fill, v_gradient_fill
from .dispersion import (
    LabeledFermiSurfaceParam,
    cut_dispersion_plot,
    fancy_dispersion,
    hv_reference_scan,
    labeled_fermi_surface,
    plot_dispersion,
    reference_scan_fermi_surface,
    scan_var_reference_plot,
)
from .dos import plot_core_levels, plot_dos
from .fermi_edge import fermi_edge_reference
from .fermi_surface import fermi_surface_slices, magnify_circular_regions_plot
from .holoviews import concat_along_phi_ui, fit_inspection, profile_view
from .movie import plot_movie
from .parameter import plot_parameter
from .spatial import plot_spatial_reference, reference_scan_spatial
from .spin import spin_colored_spectrum, spin_difference_spectrum, spin_polarized_spectrum
from .stack_plot import flat_stack_plot, offset_scatter_plot, stack_dispersion_plot
from .utils import (
    fancy_labels,
    remove_colorbars,
    savefig,
    simple_ax_grid,
)
