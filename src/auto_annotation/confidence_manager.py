from typing import Any, Dict, List, Optional
from ..utils.logger import get_logger

"""
YOLO Vision Studio - Confidence Manager

Manages confidence thresholds and scoring for auto-annotation quality control.
"""


class ConfidenceManager:
    """
    Manages confidence thresholds and scoring for auto-annotation.

    Implements confidence-based filtering and quality assessment for
    automated annotation workflows.
    """

    def __init__(self, accept_threshold: float = 0.6, reject_threshold: float = 0.2):
        self.logger = get_logger(__name__)
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold

    def evaluate_confidence(self, confidence: float) -> str:
        """
        Evaluate confidence score and return action.

        Args:
            confidence: Confidence score between 0 and 1

        Returns:
            Action string: 'accept', 'review', or 'reject'
        """
        if confidence >= self.accept_threshold:
            return "accept"
        elif confidence <= self.reject_threshold:
            return "reject"
        else:
            return "review"

    def filter_annotations_by_confidence(
        self, annotations: List[Dict[str, Any]], min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter annotations based on confidence threshold.

        Args:
            annotations: List of annotation dictionaries
            min_confidence: Minimum confidence threshold (uses accept_threshold if None)

        Returns:
            Filtered list of annotations
        """
        threshold = min_confidence or self.accept_threshold
        return [ann for ann in annotations if ann.get("confidence", 0.0) >= threshold]

    def get_confidence_statistics(
        self, annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate confidence statistics for a set of annotations.

        Args:
            annotations: List of annotation dictionaries

        Returns:
            Dictionary with confidence statistics
        """
        if not annotations:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}

        confidences = [ann.get("confidence", 0.0) for ann in annotations]

        return {
            "count": len(confidences),
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "accept_count": sum(1 for c in confidences if c >= self.accept_threshold),
            "review_count": sum(
                1
                for c in confidences
                if self.reject_threshold < c < self.accept_threshold
            ),
            "reject_count": sum(1 for c in confidences if c <= self.reject_threshold),
        }
