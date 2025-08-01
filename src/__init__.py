"""DMS - Detection Model Suite.

A comprehensive object detection pipeline with integrated authentication,
annotation tools, and model training capabilities.
"""

from importlib.metadata import PackageNotFoundError, version

from .studio import DMS

try:
    __version__ = version("dms-detection-suite")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "dev"

__author__ = "DMS Team"
__email__ = "team@dms-detection.com"
__license__ = "MIT"

# Main exports
__all__ = [
    "DMS",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
