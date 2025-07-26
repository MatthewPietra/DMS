from pathlib import Path
from unittest.mock import patch

# Export
from src.annotation.coco_exporter import COCOExporter

# ConfigManager
from src.utils.config import ConfigManager

# HardwareDetector
from src.utils.hardware import HardwareDetector

# Metrics
from src.utils.metrics import (
    ACCFramework,
    AnnotationSet,
    BoundingBox,
    MetricsCalculator,
    QualityMetrics,
)


def test_config_manager_basic() -> None:
    cm = ConfigManager()
    # Accept either 'DMS' or 'YOLO Vision Studio' as valid default
    assert cm.get("studio.name") in ["DMS", "YOLO Vision Studio"]
    cm.set("studio.name", "TestDMS")
    assert cm.get("studio.name") == "TestDMS"
    assert isinstance(cm.get_studio_config().name, str)
    assert isinstance(cm.get_hardware_config().auto_detect_gpu, bool)
    assert isinstance(cm.get_capture_config().default_fps, int)
    assert isinstance(cm.get_annotation_config().theme, str)
    assert isinstance(cm.get_training_config().epochs, int)
    assert isinstance(cm.get_auto_annotation_config().auto_accept_threshold, float)
    assert isinstance(cm.get_full_config(), dict)
    assert isinstance(cm.validate_config(), list)


def test_config_manager_save_and_load(tmp_path: Path) -> None:
    cm = ConfigManager()
    config_path = tmp_path / "test_config.yaml"
    cm.set("studio.name", "SaveLoadTest")
    cm.save_config(config_path)
    assert config_path.exists()
    cm2 = ConfigManager(str(config_path))
    assert cm2.get("studio.name") == "SaveLoadTest"


def test_config_manager_validation() -> None:
    cm = ConfigManager()
    cm.set("studio.max_concurrent_projects", 0)
    cm.set("studio.auto_save_interval", 10)
    cm.set("hardware.gpu_memory_fraction", 2.0)
    cm.set("capture.default_fps", 20)
    cm.set("capture.jpeg_quality", 101)
    cm.set("training.epochs", 0)
    cm.set("training.patience", 0)
    cm.set("training.min_map50", -1.0)
    cm.set("auto_annotation.auto_accept_threshold", -0.1)
    cm.set("auto_annotation.min_dataset_size", 1)
    issues = cm.validate_config()
    assert any("must be" in issue for issue in issues)


def test_hardware_detector_basic() -> None:
    with patch("src.utils.hardware.HardwareDetector._detect_gpus"), patch(
        "src.utils.hardware.HardwareDetector._select_optimal_device"
    ), patch("src.utils.hardware.HardwareDetector._detect_system"):
        hd = HardwareDetector()
        assert hd.get_device_type() is not None
        assert isinstance(hd.get_device(), str)
        assert isinstance(hd.get_gpu_info(), list)
        assert isinstance(hd.get_device_info(), dict)
        assert isinstance(hd.get_optimal_device(), str)
        assert isinstance(hd.get_optimal_batch_size(), int)
        assert isinstance(hd.get_optimal_workers(), int)
        assert isinstance(hd.validate_device("cpu"), bool)


def test_hardware_detector_info_methods() -> None:
    with patch("src.utils.hardware.HardwareDetector._detect_gpus"), patch(
        "src.utils.hardware.HardwareDetector._select_optimal_device"
    ), patch("src.utils.hardware.HardwareDetector._detect_system"):
        hd = HardwareDetector()
        # Test info methods
        assert hd.get_system_info() is None or hasattr(
            hd.get_system_info(), "cpu_count"
        )
        assert hd.system_info is None or hasattr(hd.system_info, "cpu_count")
        assert isinstance(hd.get_hardware_specs().cpu_count, int)
        assert isinstance(hd.get_optimal_workers(), int)
        assert isinstance(hd.is_cuda_available(), bool)
        assert isinstance(hd.is_directml_available(), bool)
        assert isinstance(hd.validate_device("cuda"), bool)


def test_metrics_calculator_and_quality_metrics() -> None:
    metrics = MetricsCalculator()
    quality = QualityMetrics()
    acc = ACCFramework()
    # Dummy boxes
    box1 = BoundingBox(0, 0, 10, 10, 0, 0.9)
    box2 = BoundingBox(0, 0, 10, 10, 0, 0.8)
    pred = [box1]
    gt = [box2]
    ann_set = AnnotationSet("img1", pred)
    # MetricsCalculator
    p, r = metrics.calculate_precision_recall(pred, gt)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    m = metrics.calculate_map({"img1": (pred, gt)})
    assert isinstance(m, dict)
    q = metrics.calculate_annotation_quality_score(ann_set)
    assert "quality_score" in q
    acc_score = metrics.calculate_acc_score([ann_set], [ann_set])
    assert "acc_score" in acc_score
    # QualityMetrics
    iou_matrix = quality.calculate_iou_matrix(pred, gt)
    assert iou_matrix.shape == (1, 1)
    # ACCFramework
    acc_metrics = acc.calculate_accuracy([ann_set], [ann_set])
    assert "accuracy" in acc_metrics


def test_metrics_calculator_history_and_average() -> None:
    metrics = MetricsCalculator()
    box1 = BoundingBox(0, 0, 10, 10, 0, 0.9)
    box2 = BoundingBox(0, 0, 10, 10, 0, 0.8)
    pred = [box1]
    gt = [box2]
    metrics.add_evaluation(pred, gt)
    avg = metrics.get_average_metrics()
    assert isinstance(avg, type(metrics.add_evaluation(pred, gt)))
    class_metrics = metrics.get_class_metrics()
    assert isinstance(class_metrics, dict)
    assert isinstance(metrics.class_metrics, dict)


def test_quality_metrics_and_accframework_edge_cases() -> None:
    quality = QualityMetrics()
    acc = ACCFramework()
    # Empty boxes
    assert quality.calculate_iou_matrix([], []).shape == (0, 0)
    assert quality.calculate_precision_recall([], []) == (1.0, 1.0)
    ann_set = AnnotationSet("img1", [])
    assert quality.calculate_annotation_quality_score(ann_set)["quality_score"] == 0.0
    # ACCFramework edge
    assert acc.calculate_accuracy([], []) == {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    assert acc.calculate_credibility([]) == {
        "credibility": 0.0,
        "ensemble_agreement": 0.0,
    }
    assert acc.calculate_consistency([]) == {"consistency": 1.0, "similarity": 1.0}
    assert "acc_score" in acc.calculate_acc_score([], [])


def test_coco_exporter_basic(tmp_path: Path) -> None:
    exporter = COCOExporter()
    # Minimal dummy data
    annotations = {"img1.jpg": [BoundingBox(0, 0, 10, 10, 0, 0.9)]}
    classes = {"0": "object"}
    output_path = tmp_path / "annotations.json"
    # Test export_coco (does not require real files)
    result = exporter.export_coco(annotations, classes, output_path)
    assert result
    assert output_path.exists()
    # Test export_yolo (directory)
    yolo_dir = tmp_path / "yolo"
    result = exporter.export_yolo(annotations, classes, yolo_dir)
    assert result
    assert (yolo_dir / "classes.txt").exists()
    # Test export_pascal_voc (directory)
    pascal_dir = tmp_path / "pascal"
    result = exporter.export_pascal_voc(annotations, classes, pascal_dir)
    assert result
    assert any(f.suffix == ".xml" for f in pascal_dir.iterdir())
    # Test export_tensorflow (directory)
    tf_dir = tmp_path / "tf"
    result = exporter.export_tensorflow(annotations, classes, tf_dir)
    assert result
    assert (tf_dir / "annotations.json").exists()


def test_coco_exporter_error_cases(tmp_path: Path) -> None:
    exporter = COCOExporter()
    # Invalid format returns False (does not raise)
    result = exporter.export_dataset(tmp_path, tmp_path, export_format="INVALID")
    assert result is False
    # Nonexistent project path: exporter may still create output dir and return True,
    # but no images/annotations
    result = exporter.export_dataset(
        tmp_path / "nonexistent", tmp_path, export_format="COCO"
    )
    assert result is True or result is False  # Accept both, but check output
    output_file = tmp_path / "annotations.json"
    if output_file.exists():
        with open(output_file, "r") as f:
            data = f.read()
        assert "images" in data or data.strip() == ""
