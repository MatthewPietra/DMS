#!/usr/bin/env python3
"""DMS Command Line Interface.

Provides command-line access to all DMS functionality including
capture, annotation, training, and project management.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from dms import __version__
from dms.utils.logger import setup_logger


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    setup_logger("dms-cli", level=level)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser.

    Returns:
        Configured argument parser
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
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
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

    return parser


def cmd_studio(args: argparse.Namespace) -> int:
    """Launch DMS Studio GUI.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    try:
        from dms.gui.main_window import main as gui_main

        return gui_main(project_path=args.project)
    except ImportError as e:
        logging.error(f"GUI not available: {e}")
        logging.error(
            "Install GUI dependencies: pip install 'dms-detection-suite[gui]'"
        )
        return 1
    except Exception as e:
        logging.error(f"Failed to launch studio: {e}")
        return 1


def cmd_capture(args: argparse.Namespace) -> int:
    """Run screen capture.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    try:
        from dms.capture import CaptureSession

        logging.info(f"Starting capture for {args.duration} seconds")
        logging.info(f"Output directory: {args.output}")

        session = CaptureSession(
            output_dir=args.output,
            fps=args.fps,
            window_title=args.window,
        )

        results = session.capture(duration=args.duration)

        logging.info(f"Capture completed: {results['images_captured']} images")
        return 0

    except Exception as e:
        logging.error(f"Capture failed: {e}")
        return 1


def cmd_train(args: argparse.Namespace) -> int:
    """Run model training.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    try:
        from dms.training import YOLOTrainer

        logging.info(f"Starting training with {args.model}")
        logging.info(f"Dataset: {args.data}")
        logging.info(f"Epochs: {args.epochs}")

        trainer = YOLOTrainer(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
        )

        results = trainer.train(
            data_path=args.data,
            epochs=args.epochs,
            output_dir=args.output,
        )

        logging.info(f"Training completed!")
        logging.info(f"Best mAP: {results.best_map:.3f}")
        logging.info(f"Model saved: {results.model_path}")

        return 0

    except Exception as e:
        logging.error(f"Training failed: {e}")
        return 1


def cmd_annotate(args: argparse.Namespace) -> int:
    """Run annotation tool.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    try:
        if args.auto:
            from dms.auto_annotation import AutoAnnotator

            if not args.model:
                logging.error("Model path required for auto-annotation")
                return 1

            logging.info("Starting auto-annotation")
            annotator = AutoAnnotator(model_path=args.model)

            results = annotator.annotate_directory(
                images_dir=args.images,
                output_dir=args.output,
                classes=args.classes,
            )

            logging.info(
                f"Auto-annotation completed: {results['images_processed']} images"
            )

        else:
            from dms.annotation import AnnotationInterface

            logging.info("Starting manual annotation interface")
            interface = AnnotationInterface(
                images_dir=args.images,
                output_dir=args.output,
                classes=args.classes,
            )

            interface.run()

        return 0

    except Exception as e:
        logging.error(f"Annotation failed: {e}")
        return 1


def cmd_project(args: argparse.Namespace) -> int:
    """Handle project management commands.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    try:
        from dms import DMS

        dms = DMS()

        if args.project_action == "create":
            project_path = dms.create_project(
                name=args.name,
                description=args.description or "",
                classes=args.classes,
            )
            logging.info(f"Created project: {project_path}")

        elif args.project_action == "list":
            projects = dms.list_projects()
            if projects:
                logging.info("Available projects:")
                for project in projects:
                    logging.info(f"  - {project['name']}: {project['path']}")
            else:
                logging.info("No projects found")

        return 0

    except Exception as e:
        logging.error(f"Project command failed: {e}")
        return 1


def cmd_export(args: argparse.Namespace) -> int:
    """Export dataset.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    try:
        from dms import DMS

        dms = DMS()

        logging.info(f"Exporting project: {args.project}")
        logging.info(f"Format: {args.format}")

        results = dms.export_dataset(
            data_path=args.project,
            output_path=args.output,
            format=args.format,
        )

        logging.info(f"Export completed: {results['output_path']}")
        return 0

    except Exception as e:
        logging.error(f"Export failed: {e}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Display system information.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    try:
        from dms.utils.system_info import get_system_info

        info = get_system_info()

        print(f"\n🔍 DMS System Information")
        print("=" * 50)
        print(f"Version: {__version__}")
        print(f"Python: {info['python_version']}")
        print(f"Platform: {info['platform']}")
        print(f"CPU: {info['cpu_info']}")
        print(f"Memory: {info['memory_info']}")
        print(f"GPU: {info['gpu_info']}")

        if args.check:
            from dms.utils.installation_check import check_installation

            score = check_installation()
            return 0 if score >= 90 else 1

        return 0

    except Exception as e:
        logging.error(f"Info command failed: {e}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command line arguments (default: sys.argv[1:])

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging
    setup_logging(args.verbose)

    # Handle no command
    if not args.command:
        parser.print_help()
        return 0

    # Command dispatch
    commands = {
        "studio": cmd_studio,
        "capture": cmd_capture,
        "train": cmd_train,
        "annotate": cmd_annotate,
        "project": cmd_project,
        "export": cmd_export,
        "info": cmd_info,
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
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
