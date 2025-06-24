"""
YOLO Vision Studio - ACC Framework

ACC (Accuracy, Credibility, Consistency) framework for quality assessment of auto-annotations.
Provides comprehensive quality metrics and validation for intelligent annotation systems.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass

from ..utils.logger import get_logger
from ..utils.config import ConfigManager
from ..utils.metrics import MetricsCalculator


@dataclass
class ACCScores:
    """ACC framework scores."""
    accuracy: float
    credibility: float
    consistency: float
    overall_score: float
    details: Dict[str, Any]


class ACCFramework:
    """ACC (Accuracy, Credibility, Consistency) quality assessment framework."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = get_logger(__name__)
        self.metrics_calculator = MetricsCalculator()
        
        # Load thresholds from config
        acc_config = self.config.get('auto_annotation.quality_control', {})
        self.accuracy_threshold = acc_config.get('accuracy_threshold', 0.90)
        self.credibility_threshold = acc_config.get('credibility_threshold', 0.85)
        self.consistency_threshold = acc_config.get('consistency_threshold', 0.80)
        
        # Historical data for consistency tracking
        self.annotation_history: Dict[str, List[Dict]] = {}
        
    def calculate_scores(self, 
                        image_path: str,
                        annotations: List[Dict[str, Any]],
                        model_name: str,
                        ground_truth: Optional[List[Dict[str, Any]]] = None) -> ACCScores:
        """Calculate ACC scores for annotations."""
        
        # Calculate individual components
        accuracy = self._calculate_accuracy(annotations, ground_truth)
        credibility = self._calculate_credibility(annotations, model_name)
        consistency = self._calculate_consistency(image_path, annotations, model_name)
        
        # Calculate overall score (weighted average)
        overall_score = (accuracy * 0.4 + credibility * 0.3 + consistency * 0.3)
        
        # Prepare detailed breakdown
        details = {
            'accuracy_details': self._get_accuracy_details(annotations, ground_truth),
            'credibility_details': self._get_credibility_details(annotations),
            'consistency_details': self._get_consistency_details(image_path, annotations),
            'thresholds': {
                'accuracy': self.accuracy_threshold,
                'credibility': self.credibility_threshold,
                'consistency': self.consistency_threshold
            },
            'passes_thresholds': {
                'accuracy': accuracy >= self.accuracy_threshold,
                'credibility': credibility >= self.credibility_threshold,
                'consistency': consistency >= self.consistency_threshold
            }
        }
        
        return ACCScores(
            accuracy=accuracy,
            credibility=credibility,
            consistency=consistency,
            overall_score=overall_score,
            details=details
        )
        
    def _calculate_accuracy(self, 
                          annotations: List[Dict[str, Any]], 
                          ground_truth: Optional[List[Dict[str, Any]]]) -> float:
        """Calculate accuracy component of ACC framework."""
        if not ground_truth:
            # If no ground truth available, use confidence-based accuracy estimation
            if not annotations:
                return 0.0
                
            confidences = [ann.get('confidence', 0.0) for ann in annotations]
            return np.mean(confidences) if confidences else 0.0
            
        # Calculate IoU-based accuracy with ground truth
        return self._calculate_iou_accuracy(annotations, ground_truth)
        
    def _calculate_iou_accuracy(self, 
                              annotations: List[Dict[str, Any]], 
                              ground_truth: List[Dict[str, Any]]) -> float:
        """Calculate accuracy based on IoU with ground truth."""
        if not annotations or not ground_truth:
            return 0.0
            
        total_iou = 0.0
        matched_pairs = 0
        
        # Match annotations with ground truth based on IoU
        for ann in annotations:
            best_iou = 0.0
            
            for gt in ground_truth:
                if ann.get('class_id') == gt.get('class_id'):
                    iou = self._calculate_bbox_iou(ann.get('bbox', []), gt.get('bbox', []))
                    best_iou = max(best_iou, iou)
                    
            if best_iou > 0.5:  # Minimum IoU threshold for match
                total_iou += best_iou
                matched_pairs += 1
                
        return total_iou / len(ground_truth) if ground_truth else 0.0
        
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
            
        # Convert from center format to corner format if needed
        def center_to_corner(bbox):
            if len(bbox) == 4:
                cx, cy, w, h = bbox
                return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
            return bbox
            
        box1 = center_to_corner(bbox1)
        box2 = center_to_corner(bbox2)
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_credibility(self, 
                             annotations: List[Dict[str, Any]], 
                             model_name: str) -> float:
        """Calculate credibility component based on model performance and confidence distribution."""
        if not annotations:
            return 0.0
            
        # Get model performance weight
        model_weight = self._get_model_credibility_weight(model_name)
        
        # Analyze confidence distribution
        confidences = [ann.get('confidence', 0.0) for ann in annotations]
        
        if not confidences:
            return 0.0
            
        # Calculate confidence statistics
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        min_conf = np.min(confidences)
        
        # Credibility factors
        confidence_factor = mean_conf  # Higher mean confidence = higher credibility
        stability_factor = max(0.0, 1.0 - std_conf)  # Lower std = higher stability
        minimum_factor = min_conf  # Higher minimum = higher credibility
        
        # Combine factors
        credibility_score = (
            confidence_factor * 0.5 + 
            stability_factor * 0.3 + 
            minimum_factor * 0.2
        ) * model_weight
        
        return min(1.0, credibility_score)
        
    def _get_model_credibility_weight(self, model_name: str) -> float:
        """Get credibility weight based on model performance history."""
        # This would be based on historical model performance
        # For now, return a default weight based on model type
        model_weights = {
            'yolov8n': 0.8,
            'yolov8s': 0.85,
            'yolov8m': 0.9,
            'yolov8l': 0.95,
            'yolov8x': 1.0,
            'ensemble': 1.0
        }
        
        return model_weights.get(model_name, 0.8)
        
    def _calculate_consistency(self, 
                             image_path: str,
                             annotations: List[Dict[str, Any]], 
                             model_name: str) -> float:
        """Calculate consistency component based on temporal stability."""
        # Store current annotations for future consistency checks
        self._store_annotation_history(image_path, annotations, model_name)
        
        # Get historical annotations for this image
        history = self.annotation_history.get(image_path, [])
        
        if len(history) < 2:
            # Not enough history for consistency check
            return 1.0  # Assume consistent until proven otherwise
            
        # Calculate consistency with recent annotations
        recent_annotations = history[-5:]  # Last 5 annotations
        consistency_scores = []
        
        for hist_entry in recent_annotations[:-1]:  # Exclude current entry
            hist_annotations = hist_entry.get('annotations', [])
            consistency_score = self._calculate_annotation_similarity(annotations, hist_annotations)
            consistency_scores.append(consistency_score)
            
        return np.mean(consistency_scores) if consistency_scores else 1.0
        
    def _store_annotation_history(self, 
                                image_path: str, 
                                annotations: List[Dict[str, Any]], 
                                model_name: str):
        """Store annotation in history for consistency tracking."""
        if image_path not in self.annotation_history:
            self.annotation_history[image_path] = []
            
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'annotations': annotations,
            'annotation_count': len(annotations)
        }
        
        self.annotation_history[image_path].append(entry)
        
        # Keep only recent history (last 10 entries)
        if len(self.annotation_history[image_path]) > 10:
            self.annotation_history[image_path] = self.annotation_history[image_path][-10:]
            
    def _calculate_annotation_similarity(self, 
                                       annotations1: List[Dict[str, Any]], 
                                       annotations2: List[Dict[str, Any]]) -> float:
        """Calculate similarity between two sets of annotations."""
        if not annotations1 and not annotations2:
            return 1.0
            
        if not annotations1 or not annotations2:
            return 0.0
            
        # Simple similarity based on count and average confidence
        count_similarity = 1.0 - abs(len(annotations1) - len(annotations2)) / max(len(annotations1), len(annotations2))
        
        conf1 = np.mean([ann.get('confidence', 0.0) for ann in annotations1])
        conf2 = np.mean([ann.get('confidence', 0.0) for ann in annotations2])
        confidence_similarity = 1.0 - abs(conf1 - conf2)
        
        # More sophisticated similarity would involve bbox matching
        return (count_similarity + confidence_similarity) / 2
        
    def _get_accuracy_details(self, 
                            annotations: List[Dict[str, Any]], 
                            ground_truth: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Get detailed accuracy breakdown."""
        if not ground_truth:
            confidences = [ann.get('confidence', 0.0) for ann in annotations]
            return {
                'method': 'confidence_based',
                'mean_confidence': np.mean(confidences) if confidences else 0.0,
                'confidence_distribution': {
                    'min': np.min(confidences) if confidences else 0.0,
                    'max': np.max(confidences) if confidences else 0.0,
                    'std': np.std(confidences) if confidences else 0.0
                }
            }
        else:
            return {
                'method': 'iou_based',
                'ground_truth_count': len(ground_truth),
                'predicted_count': len(annotations),
                'precision': len(annotations) / len(ground_truth) if ground_truth else 0.0
            }
            
    def _get_credibility_details(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed credibility breakdown."""
        confidences = [ann.get('confidence', 0.0) for ann in annotations]
        
        return {
            'annotation_count': len(annotations),
            'confidence_stats': {
                'mean': np.mean(confidences) if confidences else 0.0,
                'std': np.std(confidences) if confidences else 0.0,
                'min': np.min(confidences) if confidences else 0.0,
                'max': np.max(confidences) if confidences else 0.0
            },
            'high_confidence_ratio': len([c for c in confidences if c > 0.8]) / len(confidences) if confidences else 0.0
        }
        
    def _get_consistency_details(self, 
                               image_path: str, 
                               annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed consistency breakdown."""
        history = self.annotation_history.get(image_path, [])
        
        return {
            'history_count': len(history),
            'current_annotation_count': len(annotations),
            'has_sufficient_history': len(history) >= 2,
            'recent_counts': [entry['annotation_count'] for entry in history[-5:]]
        }
        
    def calculate_ensemble_scores(self, 
                                image_path: str,
                                model_results: Dict[str, Any]) -> ACCScores:
        """Calculate ACC scores for ensemble predictions."""
        if not model_results:
            return ACCScores(0.0, 0.0, 0.0, 0.0, {})
            
        # Extract all annotations from ensemble
        all_annotations = []
        model_scores = {}
        
        for model_name, result in model_results.items():
            annotations = result.annotations if hasattr(result, 'annotations') else result.get('annotations', [])
            all_annotations.extend(annotations)
            
            # Calculate individual model ACC scores
            model_scores[model_name] = self.calculate_scores(
                image_path, annotations, model_name
            )
            
        # Calculate ensemble-specific metrics
        ensemble_accuracy = np.mean([scores.accuracy for scores in model_scores.values()])
        ensemble_credibility = self._calculate_ensemble_credibility(model_results)
        ensemble_consistency = np.mean([scores.consistency for scores in model_scores.values()])
        
        overall_score = (ensemble_accuracy * 0.4 + ensemble_credibility * 0.3 + ensemble_consistency * 0.3)
        
        details = {
            'ensemble_size': len(model_results),
            'model_scores': {name: {
                'accuracy': scores.accuracy,
                'credibility': scores.credibility,
                'consistency': scores.consistency,
                'overall': scores.overall_score
            } for name, scores in model_scores.items()},
            'agreement_metrics': self._calculate_ensemble_agreement_metrics(model_results)
        }
        
        return ACCScores(
            accuracy=ensemble_accuracy,
            credibility=ensemble_credibility,
            consistency=ensemble_consistency,
            overall_score=overall_score,
            details=details
        )
        
    def _calculate_ensemble_credibility(self, model_results: Dict[str, Any]) -> float:
        """Calculate credibility for ensemble predictions."""
        if len(model_results) < 2:
            return 0.5  # Low credibility for single model "ensemble"
            
        # Higher credibility for more models in agreement
        model_count = len(model_results)
        base_credibility = min(1.0, model_count / 5.0)  # Max credibility at 5+ models
        
        # Bonus for model agreement (simplified)
        agreement_bonus = 0.2 if model_count >= 3 else 0.1
        
        return min(1.0, base_credibility + agreement_bonus)
        
    def _calculate_ensemble_agreement_metrics(self, model_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate agreement metrics between ensemble models."""
        if len(model_results) < 2:
            return {'agreement': 1.0}
            
        # Simplified agreement calculation
        annotation_counts = []
        avg_confidences = []
        
        for result in model_results.values():
            annotations = result.annotations if hasattr(result, 'annotations') else result.get('annotations', [])
            annotation_counts.append(len(annotations))
            
            if annotations:
                confidences = [ann.get('confidence', 0.0) for ann in annotations]
                avg_confidences.append(np.mean(confidences))
            else:
                avg_confidences.append(0.0)
                
        # Count agreement (how similar are the detection counts)
        count_std = np.std(annotation_counts) if annotation_counts else 0.0
        count_agreement = max(0.0, 1.0 - count_std / np.mean(annotation_counts)) if np.mean(annotation_counts) > 0 else 1.0
        
        # Confidence agreement
        conf_std = np.std(avg_confidences) if avg_confidences else 0.0
        conf_agreement = max(0.0, 1.0 - conf_std)
        
        overall_agreement = (count_agreement + conf_agreement) / 2
        
        return {
            'overall_agreement': overall_agreement,
            'count_agreement': count_agreement,
            'confidence_agreement': conf_agreement,
            'model_count': len(model_results)
        }
        
    def evaluate_annotation_quality(self, acc_scores: ACCScores) -> str:
        """Evaluate overall annotation quality based on ACC scores."""
        accuracy_pass = acc_scores.accuracy >= self.accuracy_threshold
        credibility_pass = acc_scores.credibility >= self.credibility_threshold
        consistency_pass = acc_scores.consistency >= self.consistency_threshold
        
        if accuracy_pass and credibility_pass and consistency_pass:
            return "excellent"
        elif acc_scores.overall_score >= 0.8:
            return "good"
        elif acc_scores.overall_score >= 0.6:
            return "acceptable"
        else:
            return "poor"
            
    def get_improvement_recommendations(self, acc_scores: ACCScores) -> List[str]:
        """Get recommendations for improving annotation quality."""
        recommendations = []
        
        if acc_scores.accuracy < self.accuracy_threshold:
            recommendations.append("Improve model training with more diverse data")
            recommendations.append("Consider using a larger/better model architecture")
            
        if acc_scores.credibility < self.credibility_threshold:
            recommendations.append("Use ensemble predictions for higher credibility")
            recommendations.append("Increase confidence thresholds for auto-acceptance")
            
        if acc_scores.consistency < self.consistency_threshold:
            recommendations.append("Ensure consistent preprocessing and input conditions")
            recommendations.append("Consider model fine-tuning for specific domain")
            
        if not recommendations:
            recommendations.append("Quality is good - consider gradual threshold increases")
            
        return recommendations 