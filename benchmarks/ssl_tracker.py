# SSL Performance Tracking for Matrix0 Benchmarks
"""
Advanced SSL (Self-Supervised Learning) performance monitoring and analysis.
Tracks SSL head effectiveness, loss convergence, and learning patterns during benchmarks.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - SSL tracking limited")


@dataclass
class SSLHeadMetrics:
    """Metrics for a single SSL head."""
    head_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    loss: float = 0.0
    learning_rate: float = 0.0
    samples_processed: int = 0
    predictions: List[float] = field(default_factory=list)
    targets: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)


@dataclass
class SSLPerformanceMetrics:
    """Comprehensive SSL performance metrics."""
    timestamp: float
    total_ssl_loss: float = 0.0
    ssl_loss_weight: float = 0.04  # Default SSL loss weight

    # Individual SSL heads
    threat_head: SSLHeadMetrics = None
    pin_head: SSLHeadMetrics = None
    fork_head: SSLHeadMetrics = None
    control_head: SSLHeadMetrics = None
    piece_head: SSLHeadMetrics = None

    # Aggregate metrics
    overall_ssl_accuracy: float = 0.0
    ssl_learning_efficiency: float = 0.0
    ssl_convergence_rate: float = 0.0
    ssl_task_balance_score: float = 0.0

    # Training dynamics
    ssl_gradient_norm: Optional[float] = None
    ssl_parameter_updates: int = 0
    ssl_memory_usage_mb: Optional[float] = None

    def __post_init__(self):
        if self.threat_head is None:
            self.threat_head = SSLHeadMetrics("threat_detection")
        if self.pin_head is None:
            self.pin_head = SSLHeadMetrics("pin_detection")
        if self.fork_head is None:
            self.fork_head = SSLHeadMetrics("fork_detection")
        if self.control_head is None:
            self.control_head = SSLHeadMetrics("control_detection")
        if self.piece_head is None:
            self.piece_head = SSLHeadMetrics("piece_recognition")


class SSLTracker:
    """Advanced SSL performance tracker for Matrix0 benchmarks."""

    def __init__(self, ssl_loss_weight: float = 0.04):
        self.ssl_loss_weight = ssl_loss_weight
        self.heads = {
            'threat': 'threat_detection',
            'pin': 'pin_detection',
            'fork': 'fork_detection',
            'control': 'control_detection',
            'piece': 'piece_recognition'
        }

        # Historical data for analysis
        self.performance_history: List[SSLPerformanceMetrics] = []
        self.baseline_metrics: Optional[SSLPerformanceMetrics] = None

        logger.info("SSL Tracker initialized with weight: {self.ssl_loss_weight}")

    def track_ssl_performance(self, model_output: Dict[str, Any],
                            ssl_targets: Dict[str, Any],
                            loss_components: Dict[str, float]) -> SSLPerformanceMetrics:
        """Track SSL performance from model outputs and targets."""

        metrics = SSLPerformanceMetrics(
            timestamp=time.time(),
            ssl_loss_weight=self.ssl_loss_weight
        )

        try:
            # Extract SSL predictions and targets
            ssl_predictions = self._extract_ssl_predictions(model_output)
            ssl_ground_truth = self._extract_ssl_targets(ssl_targets)

            # Calculate metrics for each SSL head
            head_metrics = {}
            for head_key, head_name in self.heads.items():
                if head_key in ssl_predictions and head_key in ssl_ground_truth:
                    head_metric = self._calculate_head_metrics(
                        ssl_predictions[head_key],
                        ssl_ground_truth[head_key],
                        head_name
                    )
                    head_metrics[head_key] = head_metric

                    # Update the corresponding head in metrics
                    if head_key == 'threat':
                        metrics.threat_head = head_metric
                    elif head_key == 'pin':
                        metrics.pin_head = head_metric
                    elif head_key == 'fork':
                        metrics.fork_head = head_metric
                    elif head_key == 'control':
                        metrics.control_head = head_metric
                    elif head_key == 'piece':
                        metrics.piece_head = head_metric

            # Calculate aggregate SSL metrics
            metrics.total_ssl_loss = self._calculate_total_ssl_loss(loss_components)
            metrics.overall_ssl_accuracy = self._calculate_overall_accuracy(head_metrics)
            metrics.ssl_learning_efficiency = self._calculate_learning_efficiency(head_metrics)
            metrics.ssl_convergence_rate = self._calculate_convergence_rate()
            metrics.ssl_task_balance_score = self._calculate_task_balance_score(head_metrics)

            # Track gradient and memory information if available
            if TORCH_AVAILABLE and 'ssl_grad_norm' in loss_components:
                metrics.ssl_gradient_norm = loss_components['ssl_grad_norm']

            if 'ssl_memory_mb' in loss_components:
                metrics.ssl_memory_usage_mb = loss_components['ssl_memory_mb']

            # Store in history
            self.performance_history.append(metrics)

            # Keep only recent history (last 1000 entries)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

            logger.debug(f"SSL Performance: Loss={metrics.total_ssl_loss:.4f}, "
                        f"Accuracy={metrics.overall_ssl_accuracy:.3f}, "
                        f"Efficiency={metrics.ssl_learning_efficiency:.3f}")

        except Exception as e:
            logger.error(f"Error tracking SSL performance: {e}")
            # Return basic metrics on error
            metrics.total_ssl_loss = loss_components.get('ssl_loss', 0.0)

        return metrics

    def _extract_ssl_predictions(self, model_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract SSL predictions from model output."""
        ssl_predictions = {}

        # Look for SSL heads in model output
        for head_key in self.heads.keys():
            if f'ssl_{head_key}' in model_output:
                ssl_predictions[head_key] = model_output[f'ssl_{head_key}']
            elif head_key in model_output:
                ssl_predictions[head_key] = model_output[head_key]

        return ssl_predictions

    def _extract_ssl_targets(self, ssl_targets: Dict[str, Any]) -> Dict[str, Any]:
        """Extract SSL targets from training data."""
        targets = {}

        for head_key in self.heads.keys():
            if head_key in ssl_targets:
                targets[head_key] = ssl_targets[head_key]

        return targets

    def _calculate_head_metrics(self, predictions: Any, targets: Any,
                              head_name: str) -> SSLHeadMetrics:
        """Calculate metrics for a single SSL head."""
        metrics = SSLHeadMetrics(head_name=head_name)

        try:
            if TORCH_AVAILABLE and isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()
            if TORCH_AVAILABLE and isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu().numpy()

            # Convert to numpy arrays
            if isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray):
                # For binary classification tasks (threat, pin, fork detection)
                if predictions.shape == targets.shape and len(predictions.shape) >= 1:
                    # Flatten for metric calculation
                    pred_flat = predictions.flatten()
                    target_flat = targets.flatten()

                    # Convert to binary predictions (threshold at 0.5)
                    pred_binary = (pred_flat > 0.5).astype(int)
                    target_binary = target_flat.astype(int)

                    # Calculate metrics
                    metrics.accuracy = np.mean(pred_binary == target_binary)

                    # Precision, Recall, F1 for positive class
                    if np.sum(pred_binary) > 0:
                        precision = np.sum((pred_binary == 1) & (target_binary == 1)) / np.sum(pred_binary)
                        metrics.precision = precision

                    if np.sum(target_binary) > 0:
                        recall = np.sum((pred_binary == 1) & (target_binary == 1)) / np.sum(target_binary)
                        metrics.recall = recall

                    if metrics.precision > 0 and metrics.recall > 0:
                        metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)

                    # Store sample data for analysis
                    metrics.predictions = pred_flat[:100].tolist()  # Store first 100 samples
                    metrics.targets = target_flat[:100].tolist()

                metrics.samples_processed = len(predictions.flatten()) if hasattr(predictions, 'flatten') else 0

        except Exception as e:
            logger.warning(f"Error calculating metrics for {head_name}: {e}")

        return metrics

    def _calculate_total_ssl_loss(self, loss_components: Dict[str, float]) -> float:
        """Calculate total SSL loss from components."""
        ssl_loss = 0.0

        # Sum all SSL-related losses
        for key, value in loss_components.items():
            if 'ssl' in key.lower() or key in ['threat_loss', 'pin_loss', 'fork_loss', 'control_loss', 'piece_loss']:
                ssl_loss += value

        return ssl_loss

    def _calculate_overall_accuracy(self, head_metrics: Dict[str, SSLHeadMetrics]) -> float:
        """Calculate overall SSL accuracy across all heads."""
        if not head_metrics:
            return 0.0

        accuracies = [metrics.accuracy for metrics in head_metrics.values() if metrics.accuracy > 0]
        return np.mean(accuracies) if accuracies else 0.0

    def _calculate_learning_efficiency(self, head_metrics: Dict[str, SSLHeadMetrics]) -> float:
        """Calculate SSL learning efficiency based on task balance and performance."""
        if not head_metrics:
            return 0.0

        # Efficiency is based on how well tasks are learning relative to each other
        accuracies = [metrics.accuracy for metrics in head_metrics.values()]
        f1_scores = [metrics.f1_score for metrics in head_metrics.values()]

        if not accuracies or not f1_scores:
            return 0.0

        # Efficiency = harmonic mean of accuracy and F1 balance
        accuracy_std = np.std(accuracies)
        f1_std = np.std(f1_scores)

        # Lower standard deviation means better balance (higher efficiency)
        balance_score = 1.0 / (1.0 + accuracy_std + f1_std)
        avg_performance = np.mean(accuracies + f1_scores)

        return balance_score * avg_performance

    def _calculate_convergence_rate(self) -> float:
        """Calculate SSL convergence rate based on recent loss trends."""
        if len(self.performance_history) < 10:
            return 0.0

        # Look at recent loss trend (last 10 entries)
        recent_losses = [m.total_ssl_loss for m in self.performance_history[-10:]]
        if len(recent_losses) < 2:
            return 0.0

        # Calculate linear trend (negative slope indicates convergence)
        x = np.arange(len(recent_losses))
        slope = np.polyfit(x, recent_losses, 1)[0]

        # Convert to convergence rate (negative slope = positive convergence)
        convergence_rate = max(0.0, -slope * 100)  # Scale and ensure non-negative
        return min(convergence_rate, 1.0)  # Cap at 1.0

    def _calculate_task_balance_score(self, head_metrics: Dict[str, SSLHeadMetrics]) -> float:
        """Calculate how well balanced the SSL tasks are learning."""
        if not head_metrics:
            return 0.0

        accuracies = [metrics.accuracy for metrics in head_metrics.values()]
        losses = [metrics.loss for metrics in head_metrics.values() if metrics.loss > 0]

        if not accuracies:
            return 0.0

        # Balance score based on variance in performance
        accuracy_std = np.std(accuracies)
        balance_score = 1.0 / (1.0 + accuracy_std)  # Lower variance = higher score

        return balance_score

    def get_ssl_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive SSL performance summary."""
        if not self.performance_history:
            return {"error": "No SSL performance data available"}

        recent_metrics = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history

        summary = {
            "total_measurements": len(self.performance_history),
            "recent_avg_ssl_loss": np.mean([m.total_ssl_loss for m in recent_metrics]),
            "recent_avg_accuracy": np.mean([m.overall_ssl_accuracy for m in recent_metrics]),
            "learning_efficiency": np.mean([m.ssl_learning_efficiency for m in recent_metrics]),
            "convergence_rate": recent_metrics[-1].ssl_convergence_rate if recent_metrics else 0.0,

            "head_performance": {},
            "ssl_loss_trend": [m.total_ssl_loss for m in recent_metrics[-20:]],  # Last 20 points
            "accuracy_trend": [m.overall_ssl_accuracy for m in recent_metrics[-20:]]
        }

        # Individual head performance
        heads = ['threat_head', 'pin_head', 'fork_head', 'control_head', 'piece_head']
        for head_attr in heads:
            head_data = []
            for metric in recent_metrics:
                head = getattr(metric, head_attr)
                if head and head.accuracy > 0:
                    head_data.append({
                        'accuracy': head.accuracy,
                        'f1_score': head.f1_score,
                        'loss': head.loss
                    })

            if head_data:
                summary["head_performance"][head_attr] = {
                    'avg_accuracy': np.mean([d['accuracy'] for d in head_data]),
                    'avg_f1': np.mean([d['f1_score'] for d in head_data]),
                    'count': len(head_data)
                }

        return summary

    def detect_ssl_learning_issues(self) -> List[str]:
        """Detect potential SSL learning issues."""
        issues = []

        if len(self.performance_history) < 5:
            return ["Insufficient data for SSL analysis"]

        recent = self.performance_history[-5:]

        # Check for loss not decreasing
        losses = [m.total_ssl_loss for m in recent]
        if len(losses) >= 3 and losses[-1] > losses[0] * 0.95:
            issues.append("SSL loss not decreasing - possible learning plateau")

        # Check for low accuracy
        avg_accuracy = np.mean([m.overall_ssl_accuracy for m in recent])
        if avg_accuracy < 0.3:
            issues.append(".2f")

        # Check for unbalanced learning
        balance_score = np.mean([m.ssl_task_balance_score for m in recent])
        if balance_score < 0.5:
            issues.append(".2f")

        # Check for convergence issues
        convergence_rate = np.mean([m.ssl_convergence_rate for m in recent])
        if convergence_rate < 0.1:
            issues.append(".2f")

        return issues if issues else ["SSL learning appears healthy"]

    def get_ssl_recommendations(self) -> List[str]:
        """Generate SSL performance recommendations."""
        recommendations = []

        summary = self.get_ssl_performance_summary()
        issues = self.detect_ssl_learning_issues()

        if "error" in summary:
            return ["Collect more SSL performance data before generating recommendations"]

        # Recommendations based on performance
        if summary.get('recent_avg_accuracy', 0) < 0.4:
            recommendations.append("Consider increasing SSL loss weight or adjusting learning rate")

        if summary.get('learning_efficiency', 0) < 0.5:
            recommendations.append("SSL tasks may need better balancing - consider task-specific loss weights")

        if summary.get('convergence_rate', 0) < 0.2:
            recommendations.append("SSL convergence is slow - consider curriculum learning or data augmentation")

        # Head-specific recommendations
        head_perf = summary.get('head_performance', {})
        for head_name, perf in head_perf.items():
            if perf.get('avg_accuracy', 0) < 0.3:
                recommendations.append(f"SSL head {head_name} shows low accuracy - review training data quality")

        if not recommendations:
            recommendations.append("SSL performance is within acceptable ranges")

        return recommendations


# Global SSL tracker instance
ssl_tracker = SSLTracker()
