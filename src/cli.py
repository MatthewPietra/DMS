#!/usr/bin/env python3
"""
DMS - Command Line Interface

Complete CLI for project management, training, annotation, and capture.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List

# Import core modules with error handling
try:
    from .config import config
    from .utils.logger import setup_logger, print_banner
    from .utils.hardware import get_hardware_detector
    from .studio import DMS
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure DMS is properly installed.")
    sys.exit(1)

def create_project_command(args):
    """Create a new project"""
    logger = setup_logger("cli")
    
    try:
        studio = DMS()
        project_path = studio.create_project(
            name=args.name,
            description=args.description,
            classes=args.classes.split(',') if args.classes else ['object']
        )
        logger.info(f"Project created successfully: {project_path}")
        
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        return 1
    
    return 0

def capture_command(args):
    """Start screen capture"""
    logger = setup_logger("cli")
    
    try:
        studio = DMS()
        
        # Load project if specified
        if args.project:
            studio.load_project(args.project)
        
        # Configure capture settings
        config.capture.fps = args.fps
        config.capture.monitor = args.monitor
        config.capture.window_name = args.window or ""
        
        # Start capture
        logger.info(f"Starting capture at {args.fps} FPS...")
        studio.start_capture(
            duration=args.duration,
            output_dir=args.output
        )
        
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        return 1
    
    return 0

def train_command(args):
    """Start model training"""
    logger = setup_logger("cli")
    
    try:
        studio = DMS()
        
        # Load project
        if args.project:
            studio.load_project(args.project)
        
        # Configure training
        config.training.model_architecture = args.model
        config.training.epochs = args.epochs
        config.training.learning_rate = args.lr
        
        if args.device:
            config.hardware.device = args.device
        
        if args.batch_size:
            config.hardware.batch_size = args.batch_size
        
        # Start training
        logger.info(f"Starting training with {args.model}...")
        results = studio.train_model(
            data_path=args.data,
            model_name=args.model,
            epochs=args.epochs
        )
        
        logger.info(f"Training completed. Best mAP: {results.get('best_map', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0

def annotate_command(args):
    """Start annotation interface"""
    logger = setup_logger("cli")
    
    try:
        studio = DMS()
        
        # Load project
        if args.project:
            studio.load_project(args.project)
        
        # Start annotation interface
        logger.info("Starting annotation interface...")
        studio.start_annotation(
            data_path=args.data,
            auto_annotate=args.auto
        )
        
    except Exception as e:
        logger.error(f"Annotation failed: {e}")
        return 1
    
    return 0

def auto_annotate_command(args):
    """Run auto-annotation"""
    logger = setup_logger("cli")
    
    try:
        studio = DMS()
        
        # Load project
        if args.project:
            studio.load_project(args.project)
        
        # Configure auto-annotation
        config.annotation.confidence_threshold = args.confidence
        config.annotation.review_threshold = args.review_threshold
        
        # Run auto-annotation
        logger.info("Starting auto-annotation...")
        results = studio.auto_annotate(
            data_path=args.data,
            model_path=args.model,
            output_path=args.output
        )
        
        logger.info(f"Auto-annotation completed. Processed {results.get('total_images', 0)} images")
        
    except Exception as e:
        logger.error(f"Auto-annotation failed: {e}")
        return 1
    
    return 0

def export_command(args):
    """Export dataset"""
    logger = setup_logger("cli")
    
    try:
        studio = DMS()
        
        # Load project
        if args.project:
            studio.load_project(args.project)
        
        # Export dataset
        logger.info(f"Exporting dataset to {args.format} format...")
        studio.export_dataset(
            data_path=args.data,
            output_path=args.output,
            format=args.format
        )
        
        logger.info("Dataset export completed")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1
    
    return 0

def hardware_command(args):
    """Display hardware information"""
    logger = setup_logger("cli")
    
    try:
        detector = get_hardware_detector()
        detector.print_hardware_info()
        
    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        return 1
    
    return 0

def config_command(args):
    """Manage configuration"""
    logger = setup_logger("cli")
    
    try:
        if args.show:
            # Show current configuration
            print("\nCurrent Configuration:")
            print(f"Hardware Device: {config.hardware.device}")
            print(f"Capture FPS: {config.capture.fps}")
            print(f"Training Model: {config.training.model_architecture}")
            print(f"Training Epochs: {config.training.epochs}")
            print(f"Annotation Confidence: {config.annotation.confidence_threshold}")
            
        elif args.reset:
            # Reset to defaults
            config.__init__()
            config.save()
            logger.info("Configuration reset to defaults")
            
        elif args.set:
            # Set configuration value
            key, value = args.set.split('=', 1)
            section, param = key.split('.', 1)
            
            section_obj = getattr(config, section, None)
            if section_obj and hasattr(section_obj, param):
                # Type conversion
                current_value = getattr(section_obj, param)
                if isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                elif isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes')
                
                setattr(section_obj, param, value)
                config.save()
                logger.info(f"Set {key} = {value}")
            else:
                logger.error(f"Invalid configuration key: {key}")
                return 1
        
    except Exception as e:
        logger.error(f"Configuration command failed: {e}")
        return 1
    
    return 0

def main():
    """Main CLI entry point"""
    print_banner("DMS", "Complete Object Detection Pipeline")
    
    parser = argparse.ArgumentParser(
        description="DMS - Complete Object Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--config', type=str,
                       help='Configuration file path')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create project
    create_parser = subparsers.add_parser('create', help='Create new project')
    create_parser.add_argument('name', help='Project name')
    create_parser.add_argument('--description', '-d', default='', 
                              help='Project description')
    create_parser.add_argument('--classes', '-c', 
                              help='Comma-separated class names')
    create_parser.set_defaults(func=create_project_command)
    
    # Capture
    capture_parser = subparsers.add_parser('capture', help='Start screen capture')
    capture_parser.add_argument('--project', '-p', help='Project path')
    capture_parser.add_argument('--fps', type=int, default=5, 
                               help='Capture frame rate (1-60)')
    capture_parser.add_argument('--monitor', type=int, default=0,
                               help='Monitor index')
    capture_parser.add_argument('--window', '-w', 
                               help='Window name to capture')
    capture_parser.add_argument('--duration', type=int, 
                               help='Capture duration in seconds')
    capture_parser.add_argument('--output', '-o', default='data/captured',
                               help='Output directory')
    capture_parser.set_defaults(func=capture_command)
    
    # Training
    train_parser = subparsers.add_parser('train', help='Train YOLO model')
    train_parser.add_argument('data', help='Training data path')
    train_parser.add_argument('--project', '-p', help='Project path')
    train_parser.add_argument('--model', '-m', default='yolov8n',
                             choices=['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
                                     'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                                     'yolov9c', 'yolov9e', 'yolov10n', 'yolov10s', 'yolov11n'],
                             help='YOLO model architecture')
    train_parser.add_argument('--epochs', '-e', type=int, default=100,
                             help='Training epochs')
    train_parser.add_argument('--batch-size', '-b', type=int,
                             help='Batch size (auto-detected if not specified)')
    train_parser.add_argument('--lr', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--device', choices=['auto', 'cuda', 'directml', 'cpu'],
                             help='Training device')
    train_parser.set_defaults(func=train_command)
    
    # Annotation
    annotate_parser = subparsers.add_parser('annotate', help='Start annotation interface')
    annotate_parser.add_argument('data', help='Data directory to annotate')
    annotate_parser.add_argument('--project', '-p', help='Project path')
    annotate_parser.add_argument('--auto', action='store_true',
                                help='Enable auto-annotation')
    annotate_parser.set_defaults(func=annotate_command)
    
    # Auto-annotation
    auto_annotate_parser = subparsers.add_parser('auto-annotate', 
                                                help='Run auto-annotation')
    auto_annotate_parser.add_argument('data', help='Data directory')
    auto_annotate_parser.add_argument('model', help='Model path for auto-annotation')
    auto_annotate_parser.add_argument('--project', '-p', help='Project path')
    auto_annotate_parser.add_argument('--output', '-o', default='data/auto_annotated',
                                     help='Output directory')
    auto_annotate_parser.add_argument('--confidence', type=float, default=0.6,
                                     help='Confidence threshold for auto-accept')
    auto_annotate_parser.add_argument('--review-threshold', type=float, default=0.2,
                                     help='Threshold for review queue')
    auto_annotate_parser.set_defaults(func=auto_annotate_command)
    
    # Export
    export_parser = subparsers.add_parser('export', help='Export dataset')
    export_parser.add_argument('data', help='Data directory to export')
    export_parser.add_argument('--project', '-p', help='Project path')
    export_parser.add_argument('--output', '-o', required=True,
                              help='Output directory')
    export_parser.add_argument('--format', '-f', default='coco',
                              choices=['coco', 'yolo', 'pascal', 'tensorflow'],
                              help='Export format')
    export_parser.set_defaults(func=export_command)
    
    # Hardware info
    hardware_parser = subparsers.add_parser('hardware', help='Show hardware information')
    hardware_parser.set_defaults(func=hardware_command)
    
    # Configuration
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--show', action='store_true',
                             help='Show current configuration')
    config_group.add_argument('--reset', action='store_true',
                             help='Reset configuration to defaults')
    config_group.add_argument('--set', help='Set configuration value (key=value)')
    config_parser.set_defaults(func=config_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load custom config if specified
    if args.config:
        config.load(args.config)
    
    # Set logging level
    if args.verbose:
        setup_logger("dms", "DEBUG")
    
    # Run command
    if hasattr(args, 'func'):
        try:
            return args.func(args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            return 1
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main()) 