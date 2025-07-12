"""
Quality Metrics and ACC Framework

Implements comprehensive quality assessment metrics including the
ACC (Accuracy, Credibility, Consistency) framework for auto-annotation
quality control and inter-annotator agreement tracking.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .logger import get_component_logger


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
        """Calculate Average Precision using 11-point interpolation."""
        if not precisions or not recalls:
            return 0.0

        # Sort by recall
        sorted_pairs = sorted(zip(recalls, precisions))
        recalls_sorted = [r for r, p in sorted_pairs]
        precisions_sorted = [p for r, p in sorted_pairs]

        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            # Find precisions for recalls >= t
            valid_precisions = [
                p for r, p in zip(recalls_sorted, precisions_sorted) if r >= t
            ]
            max_precision = max(valid_precisions) if valid_precisions else 0.0
            ap += max_precision

        return ap / 11.0

    def calculate_map(
        self,
        annotations_dict: Dict[str, Tuple[List[BoundingBox], List[BoundingBox]]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate mAP across multiple images and classes."""
        class_metrics = defaultdict(lambda: {"precisions": [], "recalls": []})

        # Calculate precision/recall for each image
        for image_id, (pred_boxes, gt_boxes) in annotations_dict.items():
            # Group by class
            pred_by_class = defaultdict(list)
            gt_by_class = defaultdict(list)

            for box in pred_boxes:
                pred_by_class[box.class_id].append(box)

            for box in gt_boxes:
                gt_by_class[box.class_id].append(box)

            # Calculate metrics for each class
            all_classes = set(pred_by_class.keys()) | set(gt_by_class.keys())

            for class_id in all_classes:
                pred_class_boxes = pred_by_class[class_id]
                gt_class_boxes = gt_by_class[class_id]

                precision, recall = self.calculate_precision_recall(
                    pred_class_boxes, gt_class_boxes, iou_threshold
                )

                class_metrics[class_id]["precisions"].append(precision)
                class_metrics[class_id]["recalls"].append(recall)

        # Calculate AP for each class
        class_aps = {}
        for class_id, metrics in class_metrics.items():
            ap = self.calculate_ap(metrics["precisions"], metrics["recalls"])
            class_aps[class_id] = ap

        # Calculate overall mAP
        map_score = np.mean(list(class_aps.values())) if class_aps else 0.0

        return {"mAP": map_score, "class_APs": class_aps, "num_classes": len(class_aps)}

    def calculate_annotation_quality_score(
        self, annotation_set: AnnotationSet
    ) -> Dict[str, float]:
        """Calculate quality score for an annotation set."""
        if not annotation_set.boxes:
            return {"quality_score": 0.0, "completeness": 0.0, "consistency": 1.0}

        scores = {
            "quality_score": 0.0,
            "completeness": 0.0,
            "consistency": 0.0,
            "box_quality": 0.0,
            "class_distribution": 0.0,
        }

        # Box quality assessment
        box_scores = []
        for box in annotation_set.boxes:
            # Check box validity
            if box.area > 0 and box.x1 < box.x2 and box.y1 < box.y2:
                box_score = min(1.0, box.confidence)

                # Penalize very small boxes
                if box.area < 100:  # pixels
                    box_score *= 0.8

                box_scores.append(box_score)
            else:
                box_scores.append(0.0)

        scores["box_quality"] = np.mean(box_scores) if box_scores else 0.0

        # Class distribution assessment
        class_counts = defaultdict(int)
        for box in annotation_set.boxes:
            class_counts[box.class_id] += 1

        if len(class_counts) > 0:
            # Prefer balanced class distribution
            count_values = list(class_counts.values())
            distribution_score = 1.0 - (
                np.std(count_values) / (np.mean(count_values) + 1e-6)
            )
            scores["class_distribution"] = max(0.0, min(1.0, distribution_score))

        # Overall quality score
        scores["quality_score"] = (
            scores["box_quality"] * 0.6 + scores["class_distribution"] * 0.4
        )

        return scores


class ACCFramework:
    """
    ACC (Accuracy, Credibility, Consistency) Framework

    Implements comprehensive quality assessment for auto-annotation systems
    based on accuracy against ground truth, credibility of predictions,
    and consistency across similar images.
    """

    def __init__(self):
        self.logger = get_component_logger("acc_framework")
        self.quality_metrics = QualityMetrics()
        self.history: List[Dict[str, Any]] = []

    def calculate_accuracy(
        self,
        pred_annotations: List[AnnotationSet],
        gt_annotations: List[AnnotationSet],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate accuracy component of ACC framework."""
        if len(pred_annotations) != len(gt_annotations):
            raise ValueError("Prediction and ground truth lists must have same length")

        annotations_dict = {}
        for pred, gt in zip(pred_annotations, gt_annotations):
            if pred.image_id != gt.image_id:
                raise ValueError(f"Image ID mismatch: {pred.image_id} vs {gt.image_id}")
            annotations_dict[pred.image_id] = (pred.boxes, gt.boxes)

        map_results = self.quality_metrics.calculate_map(
            annotations_dict, iou_threshold
        )

        return {
            "accuracy_score": map_results["mAP"],
            "class_accuracies": map_results["class_APs"],
            "num_classes": map_results["num_classes"],
        }

    def calculate_credibility(
        self,
        annotations: List[AnnotationSet],
        ensemble_predictions: Optional[List[List[AnnotationSet]]] = None,
    ) -> Dict[str, float]:
        """Calculate credibility component based on confidence and ensemble agreement."""
        if not annotations:
            return {"credibility_score": 0.0}

        confidence_scores = []
        agreement_scores = []

        for annotation in annotations:
            # Confidence-based credibility
            if annotation.boxes:
                confidences = [box.confidence for box in annotation.boxes]
                avg_confidence = np.mean(confidences)
                confidence_scores.append(avg_confidence)
            else:
                confidence_scores.append(0.0)

            # Ensemble agreement (if available)
            if ensemble_predictions:
                # Find ensemble predictions for this image
                image_ensembles = [
                    ensemble
                    for ensemble in ensemble_predictions
                    if any(ann.image_id == annotation.image_id for ann in ensemble)
                ]

                if image_ensembles:
                    agreement_score = self._calculate_ensemble_agreement(
                        image_ensembles[0]
                    )
                    agreement_scores.append(agreement_score)

        credibility_score = np.mean(confidence_scores)

        if agreement_scores:
            ensemble_agreement = np.mean(agreement_scores)
            credibility_score = (credibility_score + ensemble_agreement) / 2

        return {
            "credibility_score": credibility_score,
            "avg_confidence": np.mean(confidence_scores),
            "ensemble_agreement": (
                np.mean(agreement_scores) if agreement_scores else None
            ),
        }

    def calculate_consistency(
        self, annotations: List[AnnotationSet], similarity_threshold: float = 0.8
    ) -> Dict[str, float]:
        """Calculate consistency component based on temporal stability."""
        if len(annotations) < 2:
            return {"consistency_score": 1.0}

        consistency_scores = []

        # Compare consecutive annotations (assuming temporal order)
        for i in range(len(annotations) - 1):
            current = annotations[i]
            next_ann = annotations[i + 1]

            # Calculate similarity between annotations
            similarity = self._calculate_annotation_similarity(current, next_ann)
            consistency_scores.append(similarity)

        # Also check for consistency within similar images
        # (This would require image similarity calculation)

        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0

        return {
            "consistency_score": overall_consistency,
            "temporal_consistency": overall_consistency,
            "pairwise_consistencies": consistency_scores,
        }

    def calculate_acc_score(
        self,
        pred_annotations: List[AnnotationSet],
        gt_annotations: Optional[List[AnnotationSet]] = None,
        ensemble_predictions: Optional[List[List[AnnotationSet]]] = None,
    ) -> Dict[str, Any]:
        """Calculate comprehensive ACC score."""
        results = {
            "acc_score": 0.0,
            "accuracy": None,
            "credibility": None,
            "consistency": None,
        }

        # Calculate Accuracy (requires ground truth)
        if gt_annotations:
            accuracy_results = self.calculate_accuracy(pred_annotations, gt_annotations)
            results["accuracy"] = accuracy_results

        # Calculate Credibility
        credibility_results = self.calculate_credibility(
            pred_annotations, ensemble_predictions
        )
        results["credibility"] = credibility_results

        # Calculate Consistency
        consistency_results = self.calculate_consistency(pred_annotations)
        results["consistency"] = consistency_results

        # Calculate overall ACC score
        acc_components = []

        if results["accuracy"]:
            acc_components.append(results["accuracy"]["accuracy_score"])

        acc_components.append(results["credibility"]["credibility_score"])
        acc_components.append(results["consistency"]["consistency_score"])

        results["acc_score"] = np.mean(acc_components)

        # Store in history
        # import time # This line was removed by the user's edit, so it's removed here.
        # self.history.append(
        #     {
        #         "timestamp": time.time(),
        #         "results": results,
        #         "num_annotations": len(pred_annotations),
        #     }
        # )

        return results

    def _calculate_ensemble_agreement(self, ensemble: List[AnnotationSet]) -> float:
        """Calculate agreement between ensemble predictions."""
        if len(ensemble) < 2:
            return 1.0

        # Calculate pairwise IoU agreements
        agreements = []

        for i in range(len(ensemble)):
            for j in range(i + 1, len(ensemble)):
                ann1, ann2 = ensemble[i], ensemble[j]
                similarity = self._calculate_annotation_similarity(ann1, ann2)
                agreements.append(similarity)

        return np.mean(agreements) if agreements else 1.0

    def _calculate_annotation_similarity(
        self, ann1: AnnotationSet, ann2: AnnotationSet
    ) -> float:
        """Calculate similarity between two annotation sets."""
        if not ann1.boxes and not ann2.boxes:
            return 1.0

        if not ann1.boxes or not ann2.boxes:
            return 0.0

        # Calculate best matching IoU for each box
        total_similarity = 0.0
        matched_boxes = 0

        for box1 in ann1.boxes:
            best_iou = 0.0
            for box2 in ann2.boxes:
                if box1.class_id == box2.class_id:
                    iou = box1.iou(box2)
                    best_iou = max(best_iou, iou)

            if best_iou > 0.1:  # Minimum threshold for matching
                total_similarity += best_iou
                matched_boxes += 1

        # Normalize by number of boxes
        max_boxes = max(len(ann1.boxes), len(ann2.boxes))
        similarity = total_similarity / max_boxes if max_boxes > 0 else 0.0

        return similarity

    def get_quality_trend(self, window_size: int = 10) -> Dict[str, List[float]]:
        """Get quality trend over recent evaluations."""
        if len(self.history) < window_size:
            window_size = len(self.history)

        recent_history = self.history[-window_size:]

        trends = {
            "acc_scores": [],
            "accuracy_scores": [],
            "credibility_scores": [],
            "consistency_scores": [],
            "timestamps": [],
        }

        for entry in recent_history:
            results = entry["results"]
            trends["acc_scores"].append(results["acc_score"])
            trends["credibility_scores"].append(
                results["credibility"]["credibility_score"]
            )
            trends["consistency_scores"].append(
                results["consistency"]["consistency_score"]
            )
            trends["timestamps"].append(entry["timestamp"])

            if results["accuracy"]:
                trends["accuracy_scores"].append(results["accuracy"]["accuracy_score"])

        return trends


# Inter-annotator agreement functions
class MetricsCalculator:
    """
    Unified metrics calculator combining quality metrics and ACC framework.

    Provides a single interface for all annotation quality assessment needs,
    combining QualityMetrics and ACCFramework functionality.
    """

    def __init__(self):
        self.logger = get_component_logger("metrics_calculator")
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
        """Calculate mAP."""
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
    """Calculate Cohen's kappa for inter-annotator agreement."""
    if len(annotator1) != len(annotator2):
        raise ValueError("Annotator lists must have same length")

    agreements = []

    for ann1, ann2 in zip(annotator1, annotator2):
        if ann1.image_id != ann2.image_id:
            continue

        # Calculate agreement for this image
        quality_metrics = QualityMetrics()
        precision, recall = quality_metrics.calculate_precision_recall(
            ann1.boxes, ann2.boxes, iou_threshold
        )

        # Agreement is F1 score
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        agreements.append(f1)

    return np.mean(agreements) if agreements else 0.0
