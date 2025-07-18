import sys

from .window_capture import main

"""
Main entry point for the capture package.
Allows running 'python -m src.capture' without import conflicts.
"""

if __name__ == "__main__":
    sys.exit(main())
