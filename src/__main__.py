#!/usr/bin/env python3
"""
DMS Main Package Entry Point.

This module allows the entire studio to be executed as a package:
    python -m src

Provides the main studio interface with full pipeline functionality.
"""
from .studio import main

if __name__ == "__main__":
    main()
