"""
Metrics and Evaluation Tests

Test suite for object detection metrics and evaluation components.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import (
    BoundingBox,
    DetectionMetrics,
    MetricsCalculator,
    calculate_iou,
    calculate_precision_recall,
    calculate_ap,
    calculate_map,
    calculate_metrics,
    calculate_inter_annotator_agreement,
)


class TestBoundingBox(unittest.TestCase):
    """Test bounding box functionality"""

    def test_bounding_box_creation(self):
        """Test bounding box creation"""
        bbox = BoundingBox(0.1, 0.2, 0.5, 0.7, class_id=1, confidence=0.8)

        self.assertEqual(bbox.x1, 0.1)
        self.assertEqual(bbox.y1, 0.2)
        self.assertEqual(bbox.x2, 0.5)
        self.assertEqual(bbox.y2, 0.7)
        self.assertEqual(bbox.class_id, 1)
        self.assertEqual(bbox.confidence, 0.8)

    def test_bounding_box_properties(self):
        """Test bounding box computed properties"""
        bbox = BoundingBox(0.1, 0.2, 0.5, 0.8, class_id=0)

        # Test area calculation
        expected_area = (0.5 - 0.1) * (0.8 - 0.2)  # 0.4 * 0.6 = 0.24
        self.assertAlmostEqual(bbox.area, expected_area, places=6)

        # Test center calculation
        expected_center = ((0.1 + 0.5) / 2, (0.2 + 0.8) / 2)  # (0.3, 0.5)
        self.assertAlmostEqual(bbox.center[0], expected_center[0], places=6)
        self.assertAlmostEqual(bbox.center[1], expected_center[1], places=6)


class TestIoUCalculation(unittest.TestCase):
    """Test IoU calculation"""

    def test_perfect_overlap(self):
        """Test IoU with perfect overlap"""
        bbox1 = BoundingBox(0.1, 0.1, 0.5, 0.5, class_id=0)
        bbox2 = BoundingBox(0.1, 0.1, 0.5, 0.5, class_id=0)

        iou = calculate_iou(bbox1, bbox2)
        self.assertAlmostEqual(iou, 1.0, places=6)

    def test_no_overlap(self):
        """Test IoU with no overlap"""
        bbox1 = BoundingBox(0.0, 0.0, 0.2, 0.2, class_id=0)
        bbox2 = BoundingBox(0.8, 0.8, 1.0, 1.0, class_id=0)

        iou = calculate_iou(bbox1, bbox2)
        self.assertAlmostEqual(iou, 0.0, places=6)

    def test_partial_overlap(self):
        """Test IoU with partial overlap"""
        bbox1 = BoundingBox(0.0, 0.0, 0.5, 0.5, class_id=0)
        bbox2 = BoundingBox(0.25, 0.25, 0.75, 0.75, class_id=0)

        # Calculate expected IoU
        # Intersection: (0.5-0.25) * (0.5-0.25) = 0.25 * 0.25 = 0.0625
        # Union: 0.25 + 0.25 - 0.0625 = 0.4375
        # IoU: 0.0625 / 0.4375 ≈ 0.1429

        iou = calculate_iou(bbox1, bbox2)
        expected_iou = 0.0625 / 0.4375
        self.assertAlmostEqual(iou, expected_iou, places=4)


class TestPrecisionRecall(unittest.TestCase):
    """Test precision and recall calculation"""

    def test_perfect_predictions(self):
        """Test with perfect predictions"""
        predictions = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9),
            BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=1, confidence=0.8),
        ]

        ground_truths = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0),
            BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=1),
        ]

        precision, recall = calculate_precision_recall(predictions, ground_truths)

        self.assertAlmostEqual(precision, 1.0, places=6)
        self.assertAlmostEqual(recall, 1.0, places=6)

    def test_no_predictions(self):
        """Test with no predictions"""
        predictions = []
        ground_truths = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0)]

        precision, recall = calculate_precision_recall(predictions, ground_truths)

        self.assertAlmostEqual(precision, 0.0, places=6)
        self.assertAlmostEqual(recall, 0.0, places=6)

    def test_no_ground_truth(self):
        """Test with no ground truth"""
        predictions = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9)]
        ground_truths = []

        precision, recall = calculate_precision_recall(predictions, ground_truths)

        self.assertAlmostEqual(precision, 0.0, places=6)
        self.assertAlmostEqual(recall, 1.0, places=6)

    def test_false_positives(self):
        """Test with false positives"""
        predictions = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9),  # TP
            BoundingBox(0.7, 0.7, 0.9, 0.9, class_id=0, confidence=0.8),  # FP
        ]

        ground_truths = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0)]

        precision, recall = calculate_precision_recall(predictions, ground_truths)

        # Precision: 1 TP / (1 TP + 1 FP) = 0.5
        # Recall: 1 TP / 1 GT = 1.0
        self.assertAlmostEqual(precision, 0.5, places=6)
        self.assertAlmostEqual(recall, 1.0, places=6)


class TestAveragePrecision(unittest.TestCase):
    """Test Average Precision calculation"""

    def test_ap_calculation(self):
        """Test AP calculation with known data"""
        predictions = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9),
            BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=0, confidence=0.8),
            BoundingBox(0.2, 0.2, 0.4, 0.4, class_id=0, confidence=0.7),
        ]

        ground_truths = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0),
            BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=0),
        ]

        ap = calculate_ap(predictions, ground_truths, class_id=0)

        # Should be a valid AP value
        self.assertGreaterEqual(ap, 0.0)
        self.assertLessEqual(ap, 1.0)

    def test_ap_empty_predictions(self):
        """Test AP with empty predictions"""
        predictions = []
        ground_truths = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0)]

        ap = calculate_ap(predictions, ground_truths)
        self.assertAlmostEqual(ap, 0.0, places=6)

    def test_ap_empty_ground_truth(self):
        """Test AP with empty ground truth"""
        predictions = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9)]
        ground_truths = []

        ap = calculate_ap(predictions, ground_truths)
        self.assertAlmostEqual(ap, 1.0, places=6)


class TestmAP(unittest.TestCase):
    """Test mean Average Precision calculation"""

    def test_map_calculation(self):
        """Test mAP calculation"""
        predictions = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9),
            BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=1, confidence=0.8),
        ]

        ground_truths = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0),
            BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=1),
        ]

        map_results = calculate_map(predictions, ground_truths)

        self.assertIn("mAP@0.50", map_results)
        self.assertGreaterEqual(map_results["mAP@0.50"], 0.0)
        self.assertLessEqual(map_results["mAP@0.50"], 1.0)

    def test_map_multiple_thresholds(self):
        """Test mAP with multiple IoU thresholds"""
        predictions = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9)]

        ground_truths = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0)]

        thresholds = [0.5, 0.75, 0.9]
        map_results = calculate_map(predictions, ground_truths, thresholds)

        self.assertIn("mAP@0.5:0.95", map_results)
        for thresh in thresholds:
            self.assertIn(f"mAP@{thresh:.2f}", map_results)


class TestDetectionMetrics(unittest.TestCase):
    """Test DetectionMetrics dataclass"""

    def test_metrics_creation(self):
        """Test metrics creation"""
        metrics = DetectionMetrics(
            precision=0.8, recall=0.7, ap=0.75, map_50=0.65, map_50_95=0.55
        )

        self.assertEqual(metrics.precision, 0.8)
        self.assertEqual(metrics.recall, 0.7)
        self.assertEqual(metrics.ap, 0.75)
        self.assertEqual(metrics.map_50, 0.65)
        self.assertEqual(metrics.map_50_95, 0.55)

    def test_f1_score_calculation(self):
        """Test automatic F1 score calculation"""
        metrics = DetectionMetrics(precision=0.8, recall=0.6)

        # F1 = 2 * (0.8 * 0.6) / (0.8 + 0.6) = 2 * 0.48 / 1.4 ≈ 0.686
        expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        self.assertAlmostEqual(metrics.f1_score, expected_f1, places=3)


class TestMetricsCalculator(unittest.TestCase):
    """Test MetricsCalculator class"""

    def setUp(self):
        """Setup test fixtures"""
        self.calculator = MetricsCalculator()

    def test_calculator_initialization(self):
        """Test calculator initialization"""
        self.assertEqual(len(self.calculator.history), 0)
        self.assertEqual(len(self.calculator.class_metrics), 0)

    def test_add_evaluation(self):
        """Test adding evaluation"""
        predictions = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9)]
        ground_truths = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0)]

        metrics = self.calculator.add_evaluation(predictions, ground_truths)

        self.assertIsInstance(metrics, DetectionMetrics)
        self.assertEqual(len(self.calculator.history), 1)

    def test_average_metrics(self):
        """Test average metrics calculation"""
        # Add multiple evaluations
        for i in range(3):
            predictions = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9)]
            ground_truths = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0)]
            self.calculator.add_evaluation(predictions, ground_truths)

        avg_metrics = self.calculator.get_average_metrics()

        self.assertIsInstance(avg_metrics, DetectionMetrics)
        self.assertGreater(avg_metrics.precision, 0)

    def test_class_metrics(self):
        """Test per-class metrics"""
        # Add evaluations for different classes
        predictions_0 = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9)]
        ground_truths_0 = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0)]

        predictions_1 = [BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=1, confidence=0.8)]
        ground_truths_1 = [BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=1)]

        self.calculator.add_evaluation(predictions_0, ground_truths_0)
        self.calculator.add_evaluation(predictions_1, ground_truths_1)

        class_metrics = self.calculator.get_class_metrics()

        self.assertIn(0, class_metrics)
        self.assertIn(1, class_metrics)
        self.assertIsInstance(class_metrics[0], DetectionMetrics)
        self.assertIsInstance(class_metrics[1], DetectionMetrics)


class TestInterAnnotatorAgreement(unittest.TestCase):
    """Test inter-annotator agreement calculation"""

    def test_perfect_agreement(self):
        """Test perfect agreement"""
        annotations_a = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0)]
        annotations_b = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0)]

        agreement = calculate_inter_annotator_agreement(annotations_a, annotations_b)
        self.assertAlmostEqual(agreement, 1.0, places=6)

    def test_no_agreement(self):
        """Test no agreement"""
        annotations_a = [BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0)]
        annotations_b = [BoundingBox(0.7, 0.7, 0.9, 0.9, class_id=1)]

        agreement = calculate_inter_annotator_agreement(annotations_a, annotations_b)
        self.assertAlmostEqual(agreement, 0.0, places=6)

    def test_partial_agreement(self):
        """Test partial agreement"""
        annotations_a = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0),
            BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=1),
        ]
        annotations_b = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0),
            BoundingBox(0.4, 0.4, 0.6, 0.6, class_id=2),
        ]

        agreement = calculate_inter_annotator_agreement(annotations_a, annotations_b)

        # Should be between 0 and 1
        self.assertGreater(agreement, 0.0)
        self.assertLess(agreement, 1.0)


class TestMetricsIntegration(unittest.TestCase):
    """Test metrics integration with other components"""

    def test_comprehensive_metrics(self):
        """Test comprehensive metrics calculation"""
        predictions = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9),
            BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=1, confidence=0.8),
            BoundingBox(0.2, 0.2, 0.4, 0.4, class_id=0, confidence=0.7),
        ]

        ground_truths = [
            BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0),
            BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=1),
        ]

        metrics = calculate_metrics(predictions, ground_truths)

        # Verify all metrics are calculated
        self.assertIsInstance(metrics, DetectionMetrics)
        self.assertGreaterEqual(metrics.precision, 0.0)
        self.assertLessEqual(metrics.precision, 1.0)
        self.assertGreaterEqual(metrics.recall, 0.0)
        self.assertLessEqual(metrics.recall, 1.0)
        self.assertGreaterEqual(metrics.f1_score, 0.0)
        self.assertLessEqual(metrics.f1_score, 1.0)


if __name__ == "__main__":
    unittest.main()
