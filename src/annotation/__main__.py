from .annotation_interface import main
import sys

"""
Main entry point for the annotation package.
Allows running 'python -m src.annotation' without import conflicts.
"""

if __name__ == "__main__":
    sys.exit(main())
