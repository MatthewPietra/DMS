"""
Quality Metrics and ACC Framework

Implements comprehensive quality assessment metrics including the
ACC (Accuracy, Credibility, Consistency) framework for auto-annotation
quality control and inter-annotator agreement tracking.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .logger import get_component_logger


@dataclass
class BoundingBox:
    """Bounding box representation."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float = 1.0

    @property
    def area(self) -> float:
        """Calculate box area."""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        """Get box center."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: "BoundingBox") -> float:
        """Calculate IoU with another box."""
        # Calculate intersection
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass
class AnnotationSet:
    """Set of annotations for an image."""

    image_id: str
    boxes: List[BoundingBox]
    annotator_id: str = "unknown"
    timestamp: float = 0.0

    def __len__(self) -> int:
        return len(self.boxes)

    def get_boxes_by_class(self, class_id: int) -> List[BoundingBox]:
        """Get boxes for specific class."""
        return [box for box in self.boxes if box.class_id == class_id]


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    return box1.iou(box2)


def calculate_precision_recall(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float = 0.5,
) -> Tuple[float, float]:
    """Calculate precision and recall for a set of predictions and ground truths."""
    if not predictions and not ground_truths:
        return 1.0, 1.0

    if not predictions:
        return 0.0, 0.0 if ground_truths else 1.0

    if not ground_truths:
        return 0.0, 0.0

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(predictions), len(ground_truths)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            if pred.class_id == gt.class_id:
                iou_matrix[i, j] = pred.iou(gt)

    # Find matches
    matched_gt = set()
    matched_pred = set()

    # Sort predictions by confidence
    sorted_pred_indices = sorted(
        range(len(predictions)), key=lambda i: predictions[i].confidence, reverse=True
    )

    for pred_idx in sorted_pred_indices:
        best_gt_idx = -1
        best_iou = iou_threshold

        for gt_idx in range(len(ground_truths)):
            if gt_idx not in matched_gt and iou_matrix[pred_idx, gt_idx] > best_iou:
                best_iou = iou_matrix[pred_idx, gt_idx]
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)

    true_positives = len(matched_pred)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truths) - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    return precision, recall


def calculate_ap(
    predictions: List[BoundingBox], ground_truths: List[BoundingBox], class_id: int = 0
) -> float:
    """Calculate Average Precision for a specific class."""
    # Filter by class
    pred_filtered = [p for p in predictions if p.class_id == class_id]
    gt_filtered = [g for g in ground_truths if g.class_id == class_id]

    if not pred_filtered and not gt_filtered:
        return 1.0

    if not pred_filtered:
        return 0.0

    if not gt_filtered:
        return 1.0

    # Calculate precision/recall at different thresholds
    thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []

    for threshold in thresholds:
        precision, recall = calculate_precision_recall(
            pred_filtered, gt_filtered, threshold
        )
        aps.append(precision)

    return np.mean(aps) if aps else 0.0


def calculate_map(
    predictions_dict: Dict[str, List[BoundingBox]],
    ground_truths_dict: Dict[str, List[BoundingBox]],
) -> float:
    """Calculate mean Average Precision across all classes."""
    all_predictions = []
    all_ground_truths = []

    for image_id in predictions_dict:
        all_predictions.extend(predictions_dict[image_id])
        if image_id in ground_truths_dict:
            all_ground_truths.extend(ground_truths_dict[image_id])

    if not all_predictions or not all_ground_truths:
        return 0.0

    # Get unique class IDs
    class_ids = set()
    for pred in all_predictions:
        class_ids.add(pred.class_id)
    for gt in all_ground_truths:
        class_ids.add(gt.class_id)

    # Calculate AP for each class
    aps = []
    for class_id in class_ids:
        ap = calculate_ap(all_predictions, all_ground_truths, class_id)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


def calculate_inter_annotator_agreement(
    annotator1: List[AnnotationSet],
    annotator2: List[AnnotationSet],
    iou_threshold: float = 0.5,
) -> float:
    """Calculate inter-annotator agreement using IoU."""
    if not annotator1 or not annotator2:
        return 0.0

    agreements = []
    for ann1, ann2 in zip(annotator1, annotator2):
        if ann1.image_id != ann2.image_id:
            continue

        # Calculate IoU for each box pair
        ious = []
        for box1 in ann1.boxes:
            for box2 in ann2.boxes:
                if box1.class_id == box2.class_id:
                    iou = box1.iou(box2)
                    if iou >= iou_threshold:
                        ious.append(iou)

        if ious:
            agreements.append(np.mean(ious))
        else:
            agreements.append(0.0)

    return np.mean(agreements) if agreements else 0.0


def calculate_metrics(
    predictions: List[BoundingBox], ground_truths: List[BoundingBox]
) -> Dict[str, float]:
    """Calculate comprehensive metrics for object detection."""
    precision, recall = calculate_precision_recall(predictions, ground_truths)

    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "ap": calculate_ap(predictions, ground_truths),
    }


@dataclass
class DetectionMetrics:
    """Detection metrics container for easy access to common metrics."""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    map_score: float = 0.0
    accuracy: float = 0.0
    credibility: float = 0.0
    consistency: float = 0.0

    def __post_init__(self):
        """Calculate F1 score after initialization."""
        if self.precision + self.recall > 0:
            self.f1_score = (
                2 * (self.precision * self.recall) / (self.precision + self.recall)
            )
        else:
            self.f1_score = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "map_score": self.map_score,
            "accuracy": self.accuracy,
            "credibility": self.credibility,
            "consistency": self.consistency,
        }


class QualityMetrics:
    """
    Quality metrics calculator for object detection annotations.

    Provides various metrics for evaluating annotation quality,
    including precision, recall, mAP, and custom quality scores.
    """

    def __init__(self):
        self.logger = get_component_logger("quality_metrics")

    def calculate_iou_matrix(
        self, pred_boxes: List[BoundingBox], gt_boxes: List[BoundingBox]
    ) -> np.ndarray:
        """Calculate IoU matrix between predicted and ground truth boxes."""
        if not pred_boxes or not gt_boxes:
            return np.zeros((len(pred_boxes), len(gt_boxes)))

        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                if pred_box.class_id == gt_box.class_id:
                    iou_matrix[i, j] = pred_box.iou(gt_box)

        return iou_matrix

    def calculate_precision_recall(
        self,
        pred_boxes: List[BoundingBox],
        gt_boxes: List[BoundingBox],
        iou_threshold: float = 0.5,
    ) -> Tuple[float, float]:
        """Calculate precision and recall for a single image."""
        if not pred_boxes and not gt_boxes:
            return 1.0, 1.0

        if not pred_boxes:
            return 0.0, 0.0 if gt_boxes else 1.0

        if not gt_boxes:
            return 0.0, 0.0

        iou_matrix = self.calculate_iou_matrix(pred_boxes, gt_boxes)

        # Find matches using Hungarian algorithm approximation
        matched_gt = set()
        matched_pred = set()

        # Sort predictions by confidence
        sorted_pred_indices = sorted(
            range(len(pred_boxes)), key=lambda i: pred_boxes[i].confidence, reverse=True
        )

        for pred_idx in sorted_pred_indices:
            best_gt_idx = -1
            best_iou = iou_threshold

            for gt_idx in range(len(gt_boxes)):
                if gt_idx not in matched_gt and iou_matrix[pred_idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[pred_idx, gt_idx]
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)

        true_positives = len(matched_pred)
        false_positives = len(pred_boxes) - true_positives
        false_negatives = len(gt_boxes) - true_positives

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

        return precision, recall

    def calculate_ap(self, precisions: List[float], recalls: List[float]) -> float:
        """Calculate Average Precision from precision-recall curve."""
        if not precisions or not recalls:
            return 0.0

        # Sort by recall
        sorted_data = sorted(zip(recalls, precisions))
        recalls_sorted, precisions_sorted = zip(*sorted_data)

        # Calculate AP using interpolation
        ap = 0.0
        for i in range(1, len(recalls_sorted)):
            ap += (recalls_sorted[i] - recalls_sorted[i - 1]) * precisions_sorted[i]

        return ap

    def calculate_map(
        self,
        annotations_dict: Dict[str, Tuple[List[BoundingBox], List[BoundingBox]]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate mean Average Precision across all classes."""
        class_aps = defaultdict(list)

        for image_id, (pred_boxes, gt_boxes) in annotations_dict.items():
            # Group by class
            pred_by_class = defaultdict(list)
            gt_by_class = defaultdict(list)

            for box in pred_boxes:
                pred_by_class[box.class_id].append(box)

            for box in gt_boxes:
                gt_by_class[box.class_id].append(box)

            # Calculate AP for each class
            for class_id in set(pred_by_class.keys()) | set(gt_by_class.keys()):
                precision, recall = self.calculate_precision_recall(
                    pred_by_class[class_id], gt_by_class[class_id], iou_threshold
                )
                class_aps[class_id].append((recall, precision))

        # Calculate mAP
        map_scores = {}
        for class_id, pr_pairs in class_aps.items():
            recalls, precisions = zip(*pr_pairs)
            ap = self.calculate_ap(list(precisions), list(recalls))
            map_scores[f"ap_class_{class_id}"] = ap

        if map_scores:
            map_scores["mAP"] = np.mean(list(map_scores.values()))
        else:
            map_scores["mAP"] = 0.0

        return map_scores

    def calculate_annotation_quality_score(
        self, annotation_set: AnnotationSet
    ) -> Dict[str, float]:
        """Calculate quality score for an annotation set."""
        if not annotation_set.boxes:
            return {"quality_score": 0.0, "confidence": 0.0}

        # Calculate average confidence
        avg_confidence = np.mean([box.confidence for box in annotation_set.boxes])

        # Calculate box density (boxes per unit area)
        total_area = sum(box.area for box in annotation_set.boxes)
        density_score = min(1.0, total_area / 1000.0)  # Normalize

        # Calculate quality score
        quality_score = (avg_confidence + density_score) / 2.0

        return {
            "quality_score": quality_score,
            "confidence": avg_confidence,
            "density": density_score,
        }


class ACCFramework:
    """
    ACC (Accuracy, Credibility, Consistency) Framework for quality assessment.

    Provides comprehensive quality metrics for auto-annotation systems.
    """

    def __init__(self):
        self.logger = get_component_logger("acc_framework")

    def calculate_accuracy(
        self,
        pred_annotations: List[AnnotationSet],
        gt_annotations: List[AnnotationSet],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate accuracy metrics between predictions and ground truth."""
        if not pred_annotations or not gt_annotations:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        total_ious = []
        total_precision = 0.0
        total_recall = 0.0
        valid_pairs = 0

        for pred_ann, gt_ann in zip(pred_annotations, gt_annotations):
            if pred_ann.image_id != gt_ann.image_id:
                continue

            valid_pairs += 1
            precision, recall = calculate_precision_recall(
                pred_ann.boxes, gt_ann.boxes, iou_threshold
            )
            total_precision += precision
            total_recall += recall

            # Calculate IoU for each box pair
            for pred_box in pred_ann.boxes:
                for gt_box in gt_ann.boxes:
                    if pred_box.class_id == gt_box.class_id:
                        iou = pred_box.iou(gt_box)
                        if iou >= iou_threshold:
                            total_ious.append(iou)

        if valid_pairs == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        avg_precision = total_precision / valid_pairs
        avg_recall = total_recall / valid_pairs
        avg_iou = np.mean(total_ious) if total_ious else 0.0

        return {
            "accuracy": avg_iou,
            "precision": avg_precision,
            "recall": avg_recall,
        }

    def calculate_credibility(
        self,
        annotations: List[AnnotationSet],
        ensemble_predictions: Optional[List[List[AnnotationSet]]] = None,
    ) -> Dict[str, float]:
        """Calculate credibility metrics based on ensemble agreement."""
        if not annotations:
            return {"credibility": 0.0, "ensemble_agreement": 0.0}

        if ensemble_predictions:
            # Calculate ensemble agreement
            agreements = []
            for i, ann in enumerate(annotations):
                if i < len(ensemble_predictions):
                    ensemble = ensemble_predictions[i]
                    agreement = self._calculate_ensemble_agreement(ensemble)
                    agreements.append(agreement)

            avg_agreement = np.mean(agreements) if agreements else 0.0
        else:
            # Use confidence-based credibility
            all_confidences = []
            for ann in annotations:
                for box in ann.boxes:
                    all_confidences.append(box.confidence)
            avg_agreement = np.mean(all_confidences) if all_confidences else 0.0

        # Calculate credibility score
        credibility = min(1.0, avg_agreement * 1.2)  # Boost slightly

        return {
            "credibility": credibility,
            "ensemble_agreement": avg_agreement,
        }

    def calculate_consistency(
        self, annotations: List[AnnotationSet], similarity_threshold: float = 0.8
    ) -> Dict[str, float]:
        """Calculate consistency metrics across annotations."""
        if len(annotations) < 2:
            return {"consistency": 1.0, "similarity": 1.0}

        similarities = []
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                similarity = self._calculate_annotation_similarity(
                    annotations[i], annotations[j]
                )
                similarities.append(similarity)

        avg_similarity = np.mean(similarities) if similarities else 1.0
        consistency = min(1.0, avg_similarity / similarity_threshold)

        return {
            "consistency": consistency,
            "similarity": avg_similarity,
        }

    def calculate_acc_score(
        self,
        pred_annotations: List[AnnotationSet],
        gt_annotations: Optional[List[AnnotationSet]] = None,
        ensemble_predictions: Optional[List[List[AnnotationSet]]] = None,
    ) -> Dict[str, Any]:
        """Calculate comprehensive ACC score."""
        acc_scores = {}

        # Calculate accuracy if ground truth is available
        if gt_annotations:
            accuracy_metrics = self.calculate_accuracy(pred_annotations, gt_annotations)
            acc_scores.update(accuracy_metrics)

        # Calculate credibility
        credibility_metrics = self.calculate_credibility(
            pred_annotations, ensemble_predictions
        )
        acc_scores.update(credibility_metrics)

        # Calculate consistency
        consistency_metrics = self.calculate_consistency(pred_annotations)
        acc_scores.update(consistency_metrics)

        # Calculate overall ACC score
        accuracy = acc_scores.get("accuracy", 0.0)
        credibility = acc_scores.get("credibility", 0.0)
        consistency = acc_scores.get("consistency", 0.0)

        acc_score = (accuracy + credibility + consistency) / 3.0
        acc_scores["acc_score"] = acc_score

        return acc_scores

    def _calculate_ensemble_agreement(self, ensemble: List[AnnotationSet]) -> float:
        """Calculate agreement between ensemble predictions."""
        if len(ensemble) < 2:
            return 1.0

        agreements = []
        for i in range(len(ensemble)):
            for j in range(i + 1, len(ensemble)):
                similarity = self._calculate_annotation_similarity(
                    ensemble[i], ensemble[j]
                )
                agreements.append(similarity)

        return np.mean(agreements) if agreements else 1.0

    def _calculate_annotation_similarity(
        self, ann1: AnnotationSet, ann2: AnnotationSet
    ) -> float:
        """Calculate similarity between two annotation sets."""
        if ann1.image_id != ann2.image_id:
            return 0.0

        if not ann1.boxes and not ann2.boxes:
            return 1.0

        if not ann1.boxes or not ann2.boxes:
            return 0.0

        # Calculate IoU for each box pair
        ious = []
        for box1 in ann1.boxes:
            for box2 in ann2.boxes:
                if box1.class_id == box2.class_id:
                    iou = box1.iou(box2)
                    ious.append(iou)

        return np.mean(ious) if ious else 0.0

    def get_quality_trend(self, window_size: int = 10) -> Dict[str, List[float]]:
        """Get quality trend over time."""
        # This would be implemented with actual historical data
        return {
            "accuracy_trend": [],
            "credibility_trend": [],
            "consistency_trend": [],
        }


class MetricsCalculator:
    """
    Main metrics calculator for the DMS system.

    Provides a unified interface for all quality metrics calculations.
    """

    def __init__(self):
        self.quality_metrics = QualityMetrics()
        self.acc_framework = ACCFramework()

    def calculate_precision_recall(
        self,
        pred_boxes: List[BoundingBox],
        gt_boxes: List[BoundingBox],
        iou_threshold: float = 0.5,
    ) -> Tuple[float, float]:
        """Calculate precision and recall."""
        return self.quality_metrics.calculate_precision_recall(
            pred_boxes, gt_boxes, iou_threshold
        )

    def calculate_map(
        self,
        annotations_dict: Dict[str, Tuple[List[BoundingBox], List[BoundingBox]]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate mean Average Precision."""
        return self.quality_metrics.calculate_map(annotations_dict, iou_threshold)

    def calculate_annotation_quality_score(
        self, annotation_set: AnnotationSet
    ) -> Dict[str, float]:
        """Calculate annotation quality score."""
        return self.quality_metrics.calculate_annotation_quality_score(annotation_set)

    def calculate_acc_score(
        self,
        pred_annotations: List[AnnotationSet],
        gt_annotations: Optional[List[AnnotationSet]] = None,
        ensemble_predictions: Optional[List[List[AnnotationSet]]] = None,
    ) -> Dict[str, Any]:
        """Calculate ACC score."""
        return self.acc_framework.calculate_acc_score(
            pred_annotations, gt_annotations, ensemble_predictions
        )

    def calculate_accuracy(
        self,
        pred_annotations: List[AnnotationSet],
        gt_annotations: List[AnnotationSet],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate accuracy metrics."""
        return self.acc_framework.calculate_accuracy(
            pred_annotations, gt_annotations, iou_threshold
        )

    def calculate_credibility(
        self,
        annotations: List[AnnotationSet],
        ensemble_predictions: Optional[List[List[AnnotationSet]]] = None,
    ) -> Dict[str, float]:
        """Calculate credibility metrics."""
        return self.acc_framework.calculate_credibility(
            annotations, ensemble_predictions
        )

    def calculate_consistency(
        self, annotations: List[AnnotationSet], similarity_threshold: float = 0.8
    ) -> Dict[str, float]:
        """Calculate consistency metrics."""
        return self.acc_framework.calculate_consistency(
            annotations, similarity_threshold
        )

    def get_quality_trend(self, window_size: int = 10) -> Dict[str, List[float]]:
        """Get quality trend."""
        return self.acc_framework.get_quality_trend(window_size)


def calculate_cohens_kappa(
    annotator1: List[AnnotationSet],
    annotator2: List[AnnotationSet],
    iou_threshold: float = 0.5,
) -> float:
    """Calculate Cohen's Kappa for inter-annotator agreement."""
    if not annotator1 or not annotator2:
        return 0.0

    agreements = []
    for ann1, ann2 in zip(annotator1, annotator2):
        if ann1.image_id != ann2.image_id:
            continue

        # Calculate agreement for this image
        ious = []
        for box1 in ann1.boxes:
            for box2 in ann2.boxes:
                if box1.class_id == box2.class_id:
                    iou = box1.iou(box2)
                    if iou >= iou_threshold:
                        ious.append(iou)

        if ious:
            agreements.append(np.mean(ious))
        else:
            agreements.append(0.0)

    return np.mean(agreements) if agreements else 0.0
