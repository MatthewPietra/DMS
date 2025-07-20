"""
Main entry point for the annotation package.

Allows running 'python -m src.annotation' without import conflicts.
"""

import sys

from .annotation_interface import main

if __name__ == "__main__":
    sys.exit(main())
