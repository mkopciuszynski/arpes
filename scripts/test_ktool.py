from pathlib import Path

from arpes.io import load_data
from arpes.plotting.qt_ktool import ktool

data_path = (
    Path(__file__).parent.parent
    / "tests"
    / "resources"
    / "datasets"
    / "basic"
    / "main_chamber_cut_0.fits"
)

data = load_data(str(data_path.absolute()), location="ALG-MC")

ktool(data)
