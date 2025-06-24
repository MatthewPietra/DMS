"""
YOLO Vision Studio - Auto-Annotator

Intelligent auto-annotation system with:
- Confidence-based annotation acceptance (0.60/0.20 thresholds)
- ACC framework (Accuracy, Credibility, Consistency) quality assessment
- Active learning workflow for continuous improvement
- Multi-model ensemble predictions
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

try:
    import torch
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.config import ConfigManager
from ..utils.metrics import MetricsCalculator
from .acc_framework import ACCFramework
from .confidence_manager import ConfidenceManager


@dataclass
class AutoAnnotationConfig:
    """Auto-annotation configuration."""
    # Confidence thresholds
    auto_accept_threshold: float = 0.60
    human_review_threshold: float = 0.20
    auto_reject_threshold: float = 0.20
    
    # Quality control
    enable_acc_framework: bool = True
    accuracy_threshold: float = 0.90
    credibility_threshold: float = 0.85
    consistency_threshold: float = 0.80
    
    # Activation criteria
    min_dataset_size: int = 100  # images per class
    min_model_performance: float = 0.70  # mAP50
    min_class_examples: int = 50
    min_acceptance_rate: float = 0.90
    
    # Processing settings
    batch_size: int = 32
    max_concurrent_batches: int = 2
    timeout_per_image: int = 30  # seconds
    
    # Ensemble settings
    use_ensemble: bool = True
    ensemble_models: List[str] = None
    ensemble_voting: str = "weighted"  # "weighted", "majority", "average"
    
    # Additional nested fields from YAML config (optional)
    thresholds: Optional[Dict[str, Any]] = None
    quality_control: Optional[Dict[str, Any]] = None
    activation: Optional[Dict[str, Any]] = None
    processing: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Process nested configuration fields from YAML."""
        # Extract values from nested configurations if provided
        if self.thresholds:
            self.auto_accept_threshold = self.thresholds.get('auto_accept', self.auto_accept_threshold)
            self.human_review_threshold = self.thresholds.get('human_review', self.human_review_threshold)
            self.auto_reject_threshold = self.thresholds.get('auto_reject', self.auto_reject_threshold)
            
        if self.quality_control:
            self.enable_acc_framework = self.quality_control.get('enable_acc_framework', self.enable_acc_framework)
            self.accuracy_threshold = self.quality_control.get('accuracy_threshold', self.accuracy_threshold)
            self.credibility_threshold = self.quality_control.get('credibility_threshold', self.credibility_threshold)
            self.consistency_threshold = self.quality_control.get('consistency_threshold', self.consistency_threshold)
            
        if self.activation:
            self.min_dataset_size = self.activation.get('min_dataset_size', self.min_dataset_size)
            self.min_model_performance = self.activation.get('min_model_performance', self.min_model_performance)
            self.min_class_examples = self.activation.get('min_class_examples', self.min_class_examples)
            self.min_acceptance_rate = self.activation.get('min_acceptance_rate', self.min_acceptance_rate)
            
        if self.processing:
            self.batch_size = self.processing.get('batch_size', self.batch_size)
            self.max_concurrent_batches = self.processing.get('max_concurrent_batches', self.max_concurrent_batches)
            self.timeout_per_image = self.processing.get('timeout_per_image', self.timeout_per_image)


@dataclass
class AutoAnnotationResult:
    """Result of auto-annotation process."""
    image_path: str
    annotations: List[Dict[str, Any]]
    confidence_scores: List[float]
    decision: str  # "accept", "review", "reject"
    acc_scores: Dict[str, float]
    processing_time: float
    model_used: str
    ensemble_agreement: Optional[float] = None


class AutoAnnotator:
    """Intelligent auto-annotation system with quality control."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize auto-annotator."""
        self.config = config_manager
        self.logger = get_logger(__name__)
        self.metrics_calculator = MetricsCalculator()
        
        # Load configuration
        auto_config = self.config.get('auto_annotation', {})
        self.auto_config = AutoAnnotationConfig(**auto_config)
        
        # Initialize components
        self.acc_framework = ACCFramework(config_manager)
        self.confidence_manager = ConfidenceManager(config_manager)
        
        # Model management
        self.models: Dict[str, YOLO] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.active_models: List[str] = []
        
        # Statistics
        self.annotation_stats = {
            'total_processed': 0,
            'auto_accepted': 0,
            'human_review': 0,
            'auto_rejected': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }
        
        self.logger.info("AutoAnnotator initialized")
        
    def load_models(self, model_paths: Dict[str, str]) -> bool:
        """Load YOLO models for auto-annotation."""
        if not ULTRALYTICS_AVAILABLE:
            self.logger.error("Ultralytics YOLO not available")
            return False
            
        loaded_count = 0
        
        for model_name, model_path in model_paths.items():
            try:
                if not Path(model_path).exists():
                    self.logger.warning(f"Model not found: {model_path}")
                    continue
                    
                model = YOLO(model_path)
                self.models[model_name] = model
                loaded_count += 1
                
                self.logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                
        if loaded_count > 0:
            self.active_models = list(self.models.keys())
            self.logger.info(f"Loaded {loaded_count} models for auto-annotation")
            return True
        else:
            self.logger.error("No models loaded for auto-annotation")
            return False
            
    def evaluate_model_performance(self, 
                                 model_name: str,
                                 validation_data_path: str) -> Dict[str, float]:
        """Evaluate model performance on validation set."""
        if model_name not in self.models:
            raise ValueError(f"Model not loaded: {model_name}")
            
        try:
            model = self.models[model_name]
            results = model.val(data=validation_data_path)
            
            performance = {
                'map50': float(results.box.map50),
                'map': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-8)
            }
            
            self.model_performance[model_name] = performance
            self.logger.info(f"Model {model_name} performance: mAP50={performance['map50']:.3f}")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate model {model_name}: {e}")
            raise
            
    def check_activation_criteria(self, 
                                dataset_stats: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if auto-annotation can be activated."""
        issues = []
        
        # Check dataset size
        total_images = dataset_stats.get('total_images', 0)
        if total_images < self.auto_config.min_dataset_size:
            issues.append(f"Dataset too small: {total_images} < {self.auto_config.min_dataset_size}")
            
        # Check class representation
        class_counts = dataset_stats.get('class_counts', {})
        for class_name, count in class_counts.items():
            if count < self.auto_config.min_class_examples:
                issues.append(f"Insufficient examples for {class_name}: {count} < {self.auto_config.min_class_examples}")
                
        # Check model performance
        for model_name in self.active_models:
            if model_name in self.model_performance:
                map50 = self.model_performance[model_name].get('map50', 0.0)
                if map50 < self.auto_config.min_model_performance:
                    issues.append(f"Model {model_name} performance too low: {map50:.3f} < {self.auto_config.min_model_performance}")
                    
        can_activate = len(issues) == 0
        
        if can_activate:
            self.logger.info("Auto-annotation activation criteria met")
        else:
            self.logger.warning("Auto-annotation activation criteria not met:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
                
        return can_activate, issues
        
    def annotate_image(self, image_path: str) -> AutoAnnotationResult:
        """Auto-annotate a single image."""
        start_time = datetime.now()
        
        try:
            # Single model annotation
            if len(self.active_models) == 1:
                result = self._annotate_single_model(image_path, self.active_models[0])
            # Ensemble annotation
            elif self.auto_config.use_ensemble and len(self.active_models) > 1:
                result = self._annotate_ensemble(image_path)
            else:
                # Use best performing model
                best_model = self._get_best_model()
                result = self._annotate_single_model(image_path, best_model)
                
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            # Update statistics
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to annotate image {image_path}: {e}")
            raise
            
    def _annotate_single_model(self, image_path: str, model_name: str) -> AutoAnnotationResult:
        """Annotate image using single model."""
        model = self.models[model_name]
        
        # Run inference
        results = model(image_path, verbose=False)
        
        # Process results
        annotations = []
        confidence_scores = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes:
                    # Extract box data
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Convert to YOLO format (center_x, center_y, width, height)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    annotation = {
                        'class_id': class_id,
                        'bbox': [float(center_x), float(center_y), float(width), float(height)],
                        'confidence': confidence,
                        'model': model_name
                    }
                    
                    annotations.append(annotation)
                    confidence_scores.append(confidence)
                    
        # Make decision based on confidence
        decision = self._make_confidence_decision(confidence_scores)
        
        # Calculate ACC scores
        acc_scores = self.acc_framework.calculate_scores(
            image_path, annotations, model_name
        )
        
        return AutoAnnotationResult(
            image_path=image_path,
            annotations=annotations,
            confidence_scores=confidence_scores,
            decision=decision,
            acc_scores=acc_scores,
            processing_time=0.0,  # Will be set by caller
            model_used=model_name
        )
        
    def _annotate_ensemble(self, image_path: str) -> AutoAnnotationResult:
        """Annotate image using ensemble of models."""
        model_results = {}
        all_annotations = []
        
        # Get predictions from all models
        for model_name in self.active_models:
            try:
                result = self._annotate_single_model(image_path, model_name)
                model_results[model_name] = result
                all_annotations.extend(result.annotations)
            except Exception as e:
                self.logger.warning(f"Model {model_name} failed on {image_path}: {e}")
                
        if not model_results:
            raise RuntimeError("All ensemble models failed")
            
        # Combine predictions using ensemble voting
        final_annotations, final_confidences, agreement = self._combine_ensemble_predictions(
            model_results, image_path
        )
        
        # Make decision
        decision = self._make_confidence_decision(final_confidences)
        
        # Calculate ensemble ACC scores
        acc_scores = self.acc_framework.calculate_ensemble_scores(
            image_path, model_results
        )
        
        return AutoAnnotationResult(
            image_path=image_path,
            annotations=final_annotations,
            confidence_scores=final_confidences,
            decision=decision,
            acc_scores=acc_scores,
            processing_time=0.0,
            model_used="ensemble",
            ensemble_agreement=agreement
        )
        
    def _combine_ensemble_predictions(self, 
                                    model_results: Dict[str, AutoAnnotationResult],
                                    image_path: str) -> Tuple[List[Dict], List[float], float]:
        """Combine predictions from multiple models."""
        if self.auto_config.ensemble_voting == "weighted":
            return self._weighted_ensemble(model_results)
        elif self.auto_config.ensemble_voting == "majority":
            return self._majority_voting_ensemble(model_results)
        else:  # average
            return self._average_ensemble(model_results)
            
    def _weighted_ensemble(self, model_results: Dict[str, AutoAnnotationResult]) -> Tuple[List[Dict], List[float], float]:
        """Weighted ensemble based on model performance."""
        # Get model weights based on performance
        weights = {}
        total_weight = 0
        
        for model_name in model_results.keys():
            if model_name in self.model_performance:
                weight = self.model_performance[model_name].get('map50', 0.5)
                weights[model_name] = weight
                total_weight += weight
            else:
                weights[model_name] = 0.5
                total_weight += 0.5
                
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
            
        # Combine predictions (simplified implementation)
        all_annotations = []
        all_confidences = []
        
        for model_name, result in model_results.items():
            weight = weights[model_name]
            for ann in result.annotations:
                weighted_ann = ann.copy()
                weighted_ann['confidence'] *= weight
                all_annotations.append(weighted_ann)
                all_confidences.append(weighted_ann['confidence'])
                
        # Calculate agreement (simplified)
        agreement = self._calculate_ensemble_agreement(model_results)
        
        return all_annotations, all_confidences, agreement
        
    def _majority_voting_ensemble(self, model_results: Dict[str, AutoAnnotationResult]) -> Tuple[List[Dict], List[float], float]:
        """Majority voting ensemble."""
        # Simplified implementation - would need more sophisticated bbox matching
        all_annotations = []
        all_confidences = []
        
        for result in model_results.values():
            all_annotations.extend(result.annotations)
            all_confidences.extend(result.confidence_scores)
            
        agreement = self._calculate_ensemble_agreement(model_results)
        
        return all_annotations, all_confidences, agreement
        
    def _average_ensemble(self, model_results: Dict[str, AutoAnnotationResult]) -> Tuple[List[Dict], List[float], float]:
        """Average ensemble predictions."""
        # Simplified implementation
        all_annotations = []
        all_confidences = []
        
        for result in model_results.values():
            all_annotations.extend(result.annotations)
            all_confidences.extend(result.confidence_scores)
            
        agreement = self._calculate_ensemble_agreement(model_results)
        
        return all_annotations, all_confidences, agreement
        
    def _calculate_ensemble_agreement(self, model_results: Dict[str, AutoAnnotationResult]) -> float:
        """Calculate agreement between ensemble models."""
        if len(model_results) < 2:
            return 1.0
            
        # Simplified agreement calculation
        # In practice, would need sophisticated bbox matching and IoU calculations
        confidences_by_model = [result.confidence_scores for result in model_results.values()]
        
        if not confidences_by_model or not all(confidences_by_model):
            return 0.0
            
        # Calculate coefficient of variation as a measure of agreement
        all_confidences = [conf for confs in confidences_by_model for conf in confs]
        
        if not all_confidences:
            return 0.0
            
        mean_conf = np.mean(all_confidences)
        std_conf = np.std(all_confidences)
        
        if mean_conf == 0:
            return 0.0
            
        # Higher agreement = lower coefficient of variation
        cv = std_conf / mean_conf
        agreement = max(0.0, 1.0 - cv)
        
        return agreement
        
    def _make_confidence_decision(self, confidence_scores: List[float]) -> str:
        """Make decision based on confidence scores."""
        if not confidence_scores:
            return "reject"
            
        max_confidence = max(confidence_scores)
        avg_confidence = np.mean(confidence_scores)
        
        # Decision logic
        if max_confidence >= self.auto_config.auto_accept_threshold:
            return "accept"
        elif max_confidence >= self.auto_config.human_review_threshold:
            return "review"
        else:
            return "reject"
            
    def _get_best_model(self) -> str:
        """Get the best performing model."""
        if not self.model_performance:
            return self.active_models[0] if self.active_models else None
            
        best_model = None
        best_score = 0.0
        
        for model_name in self.active_models:
            if model_name in self.model_performance:
                score = self.model_performance[model_name].get('map50', 0.0)
                if score > best_score:
                    best_score = score
                    best_model = model_name
                    
        return best_model or self.active_models[0]
        
    def _update_stats(self, result: AutoAnnotationResult):
        """Update annotation statistics."""
        self.annotation_stats['total_processed'] += 1
        
        if result.decision == "accept":
            self.annotation_stats['auto_accepted'] += 1
        elif result.decision == "review":
            self.annotation_stats['human_review'] += 1
        else:
            self.annotation_stats['auto_rejected'] += 1
            
        # Update averages
        if result.confidence_scores:
            current_avg = self.annotation_stats['avg_confidence']
            new_conf = np.mean(result.confidence_scores)
            total = self.annotation_stats['total_processed']
            
            self.annotation_stats['avg_confidence'] = (
                (current_avg * (total - 1) + new_conf) / total
            )
            
        # Update processing time
        current_time = self.annotation_stats['avg_processing_time']
        total = self.annotation_stats['total_processed']
        
        self.annotation_stats['avg_processing_time'] = (
            (current_time * (total - 1) + result.processing_time) / total
        )
        
    def batch_annotate(self, 
                      image_paths: List[str],
                      output_dir: Optional[str] = None) -> List[AutoAnnotationResult]:
        """Batch annotate multiple images."""
        results = []
        
        self.logger.info(f"Starting batch annotation of {len(image_paths)} images")
        
        # Process in batches
        batch_size = self.auto_config.batch_size
        max_workers = self.auto_config.max_concurrent_batches
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batches
            futures = []
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                future = executor.submit(self._process_batch, batch)
                futures.append(future)
                
            # Collect results
            for future in as_completed(futures):
                try:
                    batch_results = future.result(timeout=self.auto_config.timeout_per_image)
                    results.extend(batch_results)
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    
        # Save results if output directory specified
        if output_dir:
            self._save_batch_results(results, output_dir)
            
        self.logger.info(f"Batch annotation completed: {len(results)} results")
        
        return results
        
    def _process_batch(self, image_paths: List[str]) -> List[AutoAnnotationResult]:
        """Process a batch of images."""
        batch_results = []
        
        for image_path in image_paths:
            try:
                result = self.annotate_image(image_path)
                batch_results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                
        return batch_results
        
    def _save_batch_results(self, results: List[AutoAnnotationResult], output_dir: str):
        """Save batch annotation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        for result in results:
            image_name = Path(result.image_path).stem
            result_file = output_path / f"{image_name}_auto_annotation.json"
            
            with open(result_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
                
        # Save summary
        summary = {
            'total_images': len(results),
            'decisions': {
                'accept': sum(1 for r in results if r.decision == "accept"),
                'review': sum(1 for r in results if r.decision == "review"),
                'reject': sum(1 for r in results if r.decision == "reject")
            },
            'avg_confidence': np.mean([
                np.mean(r.confidence_scores) for r in results 
                if r.confidence_scores
            ]) if results else 0.0,
            'avg_processing_time': np.mean([r.processing_time for r in results]),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Batch results saved to {output_dir}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get auto-annotation statistics."""
        stats = self.annotation_stats.copy()
        
        # Add model information
        stats['active_models'] = self.active_models
        stats['model_performance'] = self.model_performance
        
        # Add configuration
        stats['configuration'] = asdict(self.auto_config)
        
        return stats
        
    def reset_statistics(self):
        """Reset annotation statistics."""
        self.annotation_stats = {
            'total_processed': 0,
            'auto_accepted': 0,
            'human_review': 0,
            'auto_rejected': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }
        
        self.logger.info("Statistics reset")
        
    def cleanup(self):
        """Cleanup resources."""
        self.models.clear()
        self.model_performance.clear()
        self.active_models.clear()
        
        self.logger.info("AutoAnnotator cleanup completed") 