import os as _os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

from .agent import DataAgent
from .config import WMConfig
from .core import Calibrator, MenuCalibrator, OnlineLevel, WorldModel
from .domains import DOMAINS, Documents, EarthObservation, SensorPanels
from .service import DynaProd, PlanningSnapshot, Quote

__version__ = "0.1.0"

__all__ = ["DataAgent", "WMConfig", "Calibrator", "MenuCalibrator", "OnlineLevel", "WorldModel", "DOMAINS",
           "Documents", "EarthObservation", "SensorPanels", "DynaProd", "PlanningSnapshot", "Quote",
           "__version__"]
