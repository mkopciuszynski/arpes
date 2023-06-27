from pathlib import Path

from arpes.io import load_data
from arpes.plotting.basic_tools import bkg_tool

data_path = (
    Path(__file__).parent.parent
    / "tests"
    / "resources"
    / "datasets"
    / "basic"
    / "main_chamber_cut_0.fits"
)

data = load_data(str(data_path.absolute()))


bkg_tool(data)
