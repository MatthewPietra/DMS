#!/usr/bin/env python3
"""
YOLO Vision Studio Training Module Entry Point

This module allows the training system to be executed as a package:
    python -m src.training

Provides the same functionality as running yolo_trainer.py directly
but with proper package execution.
"""

from .yolo_trainer import main

if __name__ == "__main__":
    main() 