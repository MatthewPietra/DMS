"""Main entry point for the capture package.

Allows running 'python -m src.capture' without import conflicts.
"""

import sys

from .window_capture import main

if __name__ == "__main__":
    sys.exit(main())
