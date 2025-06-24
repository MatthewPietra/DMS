# DMS - Developer Guide

## Architecture Overview

DMS (Detection Model Suite) is built with a modular architecture:

```
src/
├── capture/          # Screen capture and image acquisition
├── annotation/       # Interactive annotation interface
├── training/         # Multi-YOLO model training pipeline
├── auto_annotation/  # Intelligent auto-labeling system
└── utils/            # Shared utilities and hardware detection
    ├── bug_fixes.py              # Bug fixes and workarounds
    ├── system_optimizer.py        # System optimization for production
    └── production_validator.py    # Production readiness validation
```

## Core API Reference

### DMS Class

The main entry point for the DMS API.

```python
from dms import DMS

studio = DMS(config_path="config/studio_config.yaml")
```

#### Methods

**Project Management**
```python
# Create new project
project_path = studio.create_project(
    name="my_project",
    description="Object detection project",
    classes=["person", "car", "bike"]
)

# Load existing project
studio.load_project("path/to/project")
```

**Capture System**
```python
# Start capture session
capture_results = studio.start_capture(
    duration=60,  # seconds
    output_dir="data/captured",
    fps=5,
    window_title="Game Window"
)
```

**Training Pipeline**
```python
# Train model
training_results = studio.train_model(
    data_path="data/train",
    model_name="yolov8n",
    epochs=100,
    batch_size=16
)
```

**Auto-Annotation**
```python
# Auto-annotate images
auto_results = studio.auto_annotate(
    data_path="data/images",
    model_path="models/best.pt",
    output_path="data/auto_annotated"
)
```

**Export System**
```python
# Export dataset
export_results = studio.export_dataset(
    data_path="data/annotated",
    output_path="exports/coco",
    format="coco"
)
```

### Configuration Management

```python
from dms.utils.config import ConfigManager

# Initialize config manager
config = ConfigManager("config/studio_config.yaml")

# Get specific configurations
hardware_config = config.get_hardware_config()
training_config = config.get_training_config()
capture_config = config.get_capture_config()

# Update configurations
config.set("hardware.device", "cuda")
config.set("training.epochs", 200)
config.save()
```

### Hardware Detection

```python
from dms.utils.hardware import HardwareDetector

detector = HardwareDetector()

# Get device information
device_type = detector.get_device_type()  # CUDA, DIRECTML, CPU
device = detector.get_device()  # "cuda:0", "directml:0", "cpu"
gpu_info = detector.get_gpu_info()

# Calculate optimal batch size
batch_size = detector.get_optimal_batch_size(
    image_size=640,
    model_size="n"
)
```

## Advanced Usage

### Custom Training Workflows

#### Multi-Stage Training
```python
from dms.training.yolo_trainer import YOLOTrainer

trainer = YOLOTrainer(config)

# Stage 1: Initial training with frozen backbone
config.training.freeze_backbone = True
config.training.epochs = 50
config.training.lr0 = 0.001

stage1_results = trainer.train_model(
    data_yaml_path="data.yaml",
    training_config=config
)

# Stage 2: Fine-tuning with unfrozen backbone
config.training.freeze_backbone = False
config.training.epochs = 100
config.training.lr0 = 0.0001

stage2_results = trainer.resume_training(
    model_path=stage1_results.model_path,
    training_config=config
)
```

#### Transfer Learning
```python
# Transfer learning setup
transfer_config = {
    'source_model': 'path/to/pretrained/model.pt',
    'freeze_layers': ['backbone.conv1', 'backbone.layer1'],
    'learning_rates': {
        'backbone': 0.0001,
        'neck': 0.001,
        'head': 0.01
    }
}

results = trainer.transfer_learning_train(
    data_yaml_path="data.yaml",
    transfer_config=transfer_config,
    epochs=100
)
```

### Custom Auto-Annotation

#### Confidence Management
```python
from dms.auto_annotation.confidence_manager import ConfidenceManager

# Custom confidence thresholds per class
class_thresholds = {
    'person': 0.8,      # High confidence for person detection
    'car': 0.6,         # Medium confidence for cars
    'bicycle': 0.7      # Medium-high for bicycles
}

confidence_manager = ConfidenceManager()
confidence_manager.set_class_thresholds(class_thresholds)
```

#### Quality-Based Auto-Annotation
```python
from dms.auto_annotation.auto_annotator import AutoAnnotator

annotator = AutoAnnotator(config)

# Custom quality assessment
def custom_quality_filter(annotations, image_path):
    quality_scores = []
    for ann in annotations:
        bbox_quality = assess_bbox_quality(ann, image_path)
        context_quality = assess_context_quality(ann, image_path)
        overall_quality = (bbox_quality + context_quality) / 2
        quality_scores.append(overall_quality)
    return quality_scores

annotator.set_quality_filter(custom_quality_filter)

# Advanced auto-annotation with quality control
results = annotator.annotate_with_quality_control(
    images=image_list,
    model_path="./models/best.pt",
    min_quality=0.7,
    max_low_quality_ratio=0.1
)
```

### Performance Optimization

#### Memory Management
```python
from dms.utils.system_optimizer import SystemOptimizer

optimizer = SystemOptimizer()

# Optimize for specific scenarios
training_opts = optimizer.optimize_for_training("large")
inference_opts = optimizer.optimize_for_inference(expected_throughput=50)

# Apply optimizations
optimizer.optimize_system_for_production()
```

#### Batch Processing
```python
from dms.utils.batch_processing import BatchProcessor

processor = BatchProcessor()

def process_images_batch(image_batch, device='cuda'):
    results = []
    for image in image_batch:
        result = process_single_image(image, device)
        results.append(result)
    return results

# Optimized batch processing
images = list(Path("./images").glob("*.jpg"))
results = processor.process_batches(
    items=images,
    process_function=process_images_batch,
    batch_size=32,
    memory_per_item_mb=150,
    device='cuda'
)
```

### Quality Assurance

#### ACC Framework
```python
from dms.auto_annotation.acc_framework import ACCFramework

acc = ACCFramework()

# Calculate ACC scores
acc_scores = acc.calculate_scores(
    image_path="image.jpg",
    annotations=annotations,
    model_name="yolov8n",
    ground_truth=ground_truth_annotations
)

# Evaluate quality
quality_level = acc.evaluate_annotation_quality(acc_scores)
recommendations = acc.get_improvement_recommendations(acc_scores)
```

#### Quality Metrics
```python
from dms.utils.metrics import QualityMetrics

metrics = QualityMetrics()

# Calculate precision and recall
precision, recall = metrics.calculate_precision_recall(
    pred_boxes=predicted_boxes,
    gt_boxes=ground_truth_boxes,
    iou_threshold=0.5
)

# Calculate mAP
map_scores = metrics.calculate_map(
    annotations_dict=annotations_dict,
    iou_threshold=0.5
)
```

## Integration Examples

### MLflow Integration
```python
import mlflow
import mlflow.pytorch
from dms import DMS

mlflow.set_experiment("dms_training")

with mlflow.start_run():
    studio = DMS()
    
    # Log parameters
    mlflow.log_param("model", "yolov8n")
    mlflow.log_param("epochs", 100)
    mlflow.log_param("batch_size", 16)
    
    # Train model
    results = studio.train_model(
        data_path="data/train",
        model_name="yolov8n",
        epochs=100
    )
    
    # Log metrics
    mlflow.log_metric("mAP", results['metrics']['mAP'])
    mlflow.log_metric("precision", results['metrics']['precision'])
    
    # Log model
    mlflow.pytorch.log_model(
        pytorch_model=results['model'],
        artifact_path="model"
    )
```

### Weights & Biases Integration
```python
import wandb
from dms.utils.callbacks import WandBCallback

wandb.init(project="dms")

# Custom callback for W&B logging
class CustomWandBCallback(WandBCallback):
    def on_epoch_end(self, epoch, metrics):
        wandb.log({
            "epoch": epoch,
            "train_loss": metrics['train_loss'],
            "val_loss": metrics['val_loss'],
            "mAP": metrics['mAP']
        })

# Use callback in training
trainer = YOLOTrainer(config)
trainer.add_callback(CustomWandBCallback())
```

## Utilities & Diagnostics

### Bug Fixes & Workarounds
```python
from dms.utils.bug_fixes import apply_all_bug_fixes

# Apply all known bug fixes
apply_all_bug_fixes()

# Check system compatibility
compatibility = check_system_compatibility()
```

### Production Validation
```python
from dms.utils.production_validator import validate_production_readiness

# Validate production readiness
report = validate_production_readiness()
print(f"Status: {report['status']}")
print(f"Issues: {report['issues']}")
print(f"Recommendations: {report['recommendations']}")
```

### System Optimization
```python
from dms.utils.system_optimizer import optimize_system_for_production

# Optimize system for production
optimization_results = optimize_system_for_production()

# Get optimization status
status = get_optimization_status()
```

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_hardware.py
python -m pytest tests/test_integration.py
python -m pytest tests/test_metrics.py
```

### Writing Tests
```python
import pytest
from dms import DMS

def test_project_creation():
    studio = DMS()
    project_path = studio.create_project("test_project")
    assert project_path.exists()
    assert (project_path / "config.yaml").exists()

def test_hardware_detection():
    from dms.utils.hardware import HardwareDetector
    detector = HardwareDetector()
    device_type = detector.get_device_type()
    assert device_type in ["cuda", "directml", "cpu"]
```

## Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd dms

# Install development dependencies
pip install -e .[dev]

# Run tests
python -m pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Add docstrings for all public functions and classes
- Write unit tests for new features

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

---
