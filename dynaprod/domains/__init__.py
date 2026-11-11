from .base import Domain, ValidationError
from .documents import Documents
from .eo import EarthObservation
from .sensors import SensorPanels

DOMAINS = {"eo": EarthObservation, "sensors": SensorPanels, "documents": Documents}

__all__ = ["Domain", "ValidationError", "Documents", "EarthObservation", "SensorPanels", "DOMAINS"]
