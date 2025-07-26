#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DMS Command Line Interface.

Provides command-line access to DMS functionality including project management,
capture, training, annotation, and export capabilities.
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from .gui.main_window import DMSMainWindow
from .studio import DMS
from .utils.logger import setup_logger
from .utils.production_validator import validate_production_readiness
from .utils.secure_subprocess import get_system_info

__version__ = "dev"  # Should be set from package metadata if available


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.

    Args:
        verbose: Enable verbose logging.
    """
    setup_logger("dms-cli")
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="dms",
        description="DMS - Detection Model Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dms studio                              # Launch GUI studio
  dms capture --duration 60              # Capture for 60 seconds
  dms train --data ./dataset             # Train model on dataset
  dms annotate --images ./images         # Start annotation tool
  dms project create "My Project"        # Create new project

For more help on a specific command, use:
  dms <command> --help
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"DMS {__version__}",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )

    # Studio command
    studio_parser = subparsers.add_parser(
        "studio",
        help="Launch DMS Studio GUI",
        description="Launch the DMS Studio graphical interface",
    )
    studio_parser.add_argument(
        "--project",
        type=Path,
        help="Project directory to open",
    )

    # Capture command
    capture_parser = subparsers.add_parser(
        "capture",
        help="Screen capture functionality",
        description="Capture screen content for dataset creation",
    )
    capture_parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=30,
        help="Capture duration in seconds (default: 30)",
    )
    capture_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./captures"),
        help="Output directory (default: ./captures)",
    )
    capture_parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second (default: 5)",
    )
    capture_parser.add_argument(
        "--window",
        type=str,
        help="Target window title (default: auto-detect)",
    )

    # Training command
    train_parser = subparsers.add_parser(
        "train",
        help="Model training functionality",
        description="Train YOLO models on datasets",
    )
    train_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training dataset",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=[
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
        ],
        help="Model architecture (default: yolov8n)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device (default: auto)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (default: auto)",
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for trained model",
    )

    # Annotation command
    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Annotation functionality",
        description="Annotate images manually or automatically",
    )
    annotate_parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to images directory",
    )
    annotate_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for annotations",
    )
    annotate_parser.add_argument(
        "--auto",
        action="store_true",
        help="Enable auto-annotation",
    )
    annotate_parser.add_argument(
        "--model",
        type=Path,
        help="Model for auto-annotation",
    )
    annotate_parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        help="Class names for annotation",
    )

    # Project management command
    project_parser = subparsers.add_parser(
        "project",
        help="Project management",
        description="Create and manage DMS projects",
    )
    project_subparsers = project_parser.add_subparsers(
        dest="project_action",
        help="Project actions",
        metavar="ACTION",
    )

    # Create project
    create_parser = project_subparsers.add_parser(
        "create",
        help="Create new project",
    )
    create_parser.add_argument(
        "name",
        help="Project name",
    )
    create_parser.add_argument(
        "--description",
        help="Project description",
    )
    create_parser.add_argument(
        "--classes",
        nargs="+",
        default=["object"],
        help="Object classes (default: object)",
    )

    # List projects
    project_subparsers.add_parser(
        "list",
        help="List existing projects",
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export datasets",
        description="Export datasets in various formats",
    )
    export_parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Project directory to export",
    )
    export_parser.add_argument(
        "--format",
        choices=["coco", "yolo", "pascal"],
        default="coco",
        help="Export format (default: coco)",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="System information",
        description="Display system and installation information",
    )
    info_parser.add_argument(
        "--check",
        action="store_true",
        help="Run installation check",
    )

    # Hardware command
    subparsers.add_parser(
        "hardware",
        help="Hardware information",
        description="Display hardware detection information",
    )

    return parser


def cmd_studio(args: argparse.Namespace) -> int:
    """Launch DMS Studio GUI.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        int: Exit code.
    """
    try:
        # Launch GUI using DMSMainWindow directly
        import sys

        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        # Set application properties
        app.setApplicationName("DMS")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("DMS Team")
        # Create and show main window
        window = DMSMainWindow()
        # Load project if specified
        if hasattr(args, "project") and args.project:
            # Implement project loading in DMSMainWindow
            project_path = Path(args.project)
            if project_path.exists():
                # Check if it's a valid project directory
                config_file = project_path / "config.json"
                if config_file.exists():
                    try:
                        # Set the current project in the main window
                        window.set_current_project(str(project_path))
                        logging.info(f"Loaded project: {project_path.name}")
                    except Exception as e:
                        logging.warning(f"Failed to load project {args.project}: {e}")
                else:
                    logging.warning(
                        f"Project directory {args.project} does not contain a "
                        f"valid config.json"
                    )
            else:
                logging.warning(f"Project directory {args.project} does not exist")
        window.show()
        # Start event loop
        return app.exec()
    except Exception as e:
        print(f"Error launching DMS Studio: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_capture(args: argparse.Namespace) -> int:
    """Run screen capture.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        int: Exit code.
    """
    try:
        dms = DMS()
        logging.info(f"Starting capture for {args.duration} seconds")
        logging.info(f"Output directory: {args.output}")

        # Use DMS interface for capture with proper parameters
        results = dms.start_capture(duration=args.duration, output_dir=str(args.output))

        logging.info(f"Capture completed: {results.get('session_id', 'N/A')}")
        logging.info(f"Windows found: {results.get('windows_found', 0)}")
        logging.info(f"Output directory: {results.get('output_dir', 'N/A')}")
        return 0
    except Exception as e:
        logging.error(f"Capture failed: {e}")
        return 1


def cmd_train(args: argparse.Namespace) -> int:
    """Run model training.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        int: Exit code.
    """
    try:
        dms = DMS()
        logging.info(f"Starting training with {args.model}")
        logging.info(f"Dataset: {args.data}")
        logging.info(f"Epochs: {args.epochs}")

        # Use DMS interface for training with proper parameters
        results = dms.train_model(
            data_path=str(args.data), model_name=args.model, epochs=args.epochs
        )

        logging.info("Training completed!")
        logging.info(f"Best mAP50: {results.get('best_map50', 'N/A')}")
        logging.info(f"Model saved: {results.get('model_path', 'N/A')}")
        logging.info(f"Training time: {results.get('training_time', 'N/A')}")
        logging.info(f"Epochs completed: {results.get('epochs_completed', 'N/A')}")
        return 0
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return 1


def cmd_annotate(args: argparse.Namespace) -> int:
    """Run annotation tool.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        int: Exit code.
    """
    try:
        dms = DMS()

        if args.auto:
            if not args.model:
                logging.error("Model path required for auto-annotation")
                return 1

            logging.info("Starting auto-annotation")
            output_path = str(args.output) if args.output else "data/auto_annotated"

            results = dms.auto_annotate(
                data_path=str(args.images),
                model_path=str(args.model),
                output_path=output_path,
            )

            logging.info(
                f"Auto-annotation completed: "
                f"{results.get('images_processed', 'N/A')} images"
            )
            logging.info(f"Output path: {results.get('output_path', 'N/A')}")
        else:
            logging.info("Starting manual annotation interface")
            dms.start_annotation(str(args.images))

        return 0
    except Exception as e:
        logging.error(f"Annotation failed: {e}")
        return 1


def cmd_project(args: argparse.Namespace) -> int:
    """Handle project management commands.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        int: Exit code.
    """
    try:
        dms = DMS()

        if args.project_action == "create":
            project_path = dms.create_project(
                name=args.name,
                description=args.description or "",
                classes=args.classes,
            )
            logging.info(f"Created project: {project_path}")

        elif args.project_action == "list":
            # List projects by scanning the data/projects directory
            projects_dir = Path("data/projects")
            if not projects_dir.exists():
                logging.info("No projects directory found")
                return 0

            projects = []
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    config_file = project_dir / "config.json"
                    if config_file.exists():
                        try:
                            import json

                            with open(config_file, "r") as f:
                                config = json.load(f)
                            projects.append(
                                {
                                    "name": config.get("name", project_dir.name),
                                    "path": str(project_dir),
                                    "description": config.get("description", ""),
                                    "classes": config.get("classes", []),
                                    "created_at": config.get("created_at", ""),
                                }
                            )
                        except Exception as e:
                            logging.warning(
                                f"Error reading project config {config_file}: {e}"
                            )
                            projects.append(
                                {
                                    "name": project_dir.name,
                                    "path": str(project_dir),
                                    "description": "Error reading config",
                                    "classes": [],
                                    "created_at": "",
                                }
                            )

            if projects:
                logging.info("Available projects:")
                for project in projects:
                    logging.info(f"  - {project['name']}: {project['path']}")
                    if project["description"]:
                        logging.info(f"    Description: {project['description']}")
                    if project["classes"]:
                        logging.info(f"    Classes: {', '.join(project['classes'])}")
                    if project["created_at"]:
                        logging.info(f"    Created: {project['created_at']}")
            else:
                logging.info("No projects found")

        return 0
    except Exception as e:
        logging.error(f"Project command failed: {e}")
        return 1


def cmd_export(args: argparse.Namespace) -> int:
    """Export dataset.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        int: Exit code.
    """
    try:
        dms = DMS()
        logging.info(f"Exporting project: {args.project}")
        logging.info(f"Format: {args.format}")

        # Ensure output_path is always a string
        output_path = str(args.output) if args.output else "./export"

        results = dms.export_dataset(
            data_path=str(args.project),
            output_path=output_path,
            format=args.format,
        )

        logging.info(f"Export completed: {results.get('output_path', 'N/A')}")
        return 0
    except Exception as e:
        logging.error(f"Export failed: {e}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Display system information.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        int: Exit code.
    """
    try:
        success, stdout, stderr = get_system_info()
        print("\n🔍 DMS System Information")
        print("=" * 50)
        print(f"Version: {__version__}")
        print(stdout)

        if args.check:
            report = validate_production_readiness()
            status = report.get("status", "FAIL")
            print(f"Production readiness: {status}")
            if status != "PASS":
                print("Issues:")
                for issue in report.get("issues", []):
                    print(f"  - {issue}")
                print("Recommendations:")
                for rec in report.get("recommendations", []):
                    print(f"  - {rec}")
            return 0 if status == "PASS" else 1
        return 0
    except Exception as e:
        logging.error(f"Info command failed: {e}")
        return 1


def show_hardware_info() -> None:
    """Display hardware information."""
    try:
        from .utils.hardware import get_hardware_detector

        detector = get_hardware_detector()
        specs = detector.detect_hardware()

        print("\n🔧 Hardware Information")
        print("=" * 50)
        print(f"Device Type: {specs.device_type}")
        print(f"Optimal Device: {specs.optimal_device}")
        print(f"CPU Count: {specs.cpu_count}")
        print(f"GPU Count: {len(specs.gpus)}")

        for i, gpu in enumerate(specs.gpus):
            print(f"GPU {i}: {gpu.name} ({gpu.memory_total}MB)")

    except Exception as e:
        logging.error(f"Hardware info failed: {e}")


def cmd_hardware(args: argparse.Namespace) -> int:
    """Display hardware information.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        int: Exit code.
    """
    try:
        show_hardware_info()
        return 0
    except Exception as e:
        logging.error(f"Hardware command failed: {e}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Run the main CLI application.

    Args:
        argv: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if not args.command:
        parser.print_help()
        return 0
    commands = {
        "studio": cmd_studio,
        "capture": cmd_capture,
        "train": cmd_train,
        "annotate": cmd_annotate,
        "project": cmd_project,
        "export": cmd_export,
        "info": cmd_info,
        "hardware": cmd_hardware,
    }
    try:
        return commands[args.command](args)
    except KeyError:
        logging.error(f"Unknown command: {args.command}")
        return 1
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
