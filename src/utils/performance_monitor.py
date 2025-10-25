"""
Performance Monitoring Module for Japanese Tokenizer

This module provides comprehensive performance monitoring, alerting,
and optimization recommendations based on logged metrics.
"""

import time
import threading
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

# Import logging utilities
try:
    from .logger import get_logger, log_performance
except ImportError:
    # Fallback for direct execution
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def log_performance(logger, operation, duration, details=None):
        logger.info(f"Performance: {operation} took {duration:.4f}s")


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    metrics: Dict[str, Any]
    recommendations: List[str]


class PerformanceMonitor:
    """
    Comprehensive performance monitoring for the Japanese tokenizer.
    
    Tracks metrics, identifies bottlenecks, and provides optimization
    recommendations based on real-time performance data.
    """
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        self.logger = get_logger('performance_monitor')
        
        # Performance thresholds
        self.thresholds = alert_thresholds or {
            'max_processing_time_ms': 100,      # 100ms per sentence
            'min_cache_hit_rate': 0.85,         # 85% cache hit rate
            'max_memory_usage_mb': 500,         # 500MB memory usage
            'max_cpu_usage_percent': 80,        # 80% CPU usage
            'min_confidence_threshold': 0.7     # 70% confidence threshold
        }
        
        # Performance metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = deque(maxlen=100)
        self.performance_baselines = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Monitoring state
        self.monitoring_active = False
        self._monitor_thread = None
        
        self.logger.info("PerformanceMonitor initialized")
    
    def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"Performance monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def record_metric(self, metric_name: str, value: float, 
                     timestamp: Optional[datetime] = None) -> None:
        """Record a performance metric."""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            self.metrics_history[metric_name].append({
                'value': value,
                'timestamp': timestamp
            })
    
    def record_tokenization_metrics(self, processing_time: float, 
                                  token_count: int, unknown_count: int,
                                  cache_hit_rate: float) -> None:
        """Record tokenization-specific metrics."""
        timestamp = datetime.now()
        
        # Record individual metrics
        self.record_metric('processing_time_ms', processing_time * 1000, timestamp)
        self.record_metric('tokens_per_second', token_count / max(processing_time, 0.001), timestamp)
        self.record_metric('unknown_word_rate', unknown_count / max(token_count, 1), timestamp)
        self.record_metric('cache_hit_rate', cache_hit_rate, timestamp)
        
        # Check for performance issues
        self._check_performance_issues(processing_time, cache_hit_rate, timestamp)
    
    def _check_performance_issues(self, processing_time: float, 
                                cache_hit_rate: float, timestamp: datetime) -> None:
        """Check for performance issues and create alerts."""
        issues = []
        
        # Check processing time
        if processing_time * 1000 > self.thresholds['max_processing_time_ms']:
            issues.append(PerformanceAlert(
                alert_type='slow_processing',
                severity='medium',
                message=f'Processing time {processing_time*1000:.1f}ms exceeds threshold',
                timestamp=timestamp,
                metrics={'processing_time_ms': processing_time * 1000},
                recommendations=[
                    'Consider enabling batch processing for large texts',
                    'Check cache hit rate and optimize cache size',
                    'Review confidence thresholds for early termination'
                ]
            ))
        
        # Check cache hit rate
        if cache_hit_rate < self.thresholds['min_cache_hit_rate']:
            issues.append(PerformanceAlert(
                alert_type='low_cache_hit_rate',
                severity='high',
                message=f'Cache hit rate {cache_hit_rate:.2%} below threshold',
                timestamp=timestamp,
                metrics={'cache_hit_rate': cache_hit_rate},
                recommendations=[
                    'Increase L1 cache capacity',
                    'Optimize L2 cache content based on access patterns',
                    'Review cache eviction policies'
                ]
            ))
        
        # Add alerts to history
        for issue in issues:
            with self._lock:
                self.alerts.append(issue)
            self.logger.warning(f"Performance alert: {issue.message}")
    
    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._analyze_performance_trends()
                self._check_system_resources()
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _analyze_performance_trends(self) -> None:
        """Analyze performance trends and detect anomalies."""
        with self._lock:
            # Analyze recent performance trends
            recent_metrics = {}
            for metric_name, history in self.metrics_history.items():
                if len(history) >= 10:  # Need sufficient data
                    recent_values = [entry['value'] for entry in list(history)[-10:]]
                    recent_metrics[metric_name] = {
                        'average': sum(recent_values) / len(recent_values),
                        'trend': self._calculate_trend(recent_values),
                        'volatility': self._calculate_volatility(recent_values)
                    }
            
            # Detect performance degradation
            for metric_name, stats in recent_metrics.items():
                if stats['trend'] < -0.1:  # 10% degradation
                    self._create_trend_alert(metric_name, stats)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate performance trend (positive = improving, negative = degrading)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        y = values
        
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate performance volatility."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _create_trend_alert(self, metric_name: str, stats: Dict[str, float]) -> None:
        """Create alert for performance trend degradation."""
        alert = PerformanceAlert(
            alert_type='performance_degradation',
            severity='medium',
            message=f'Performance degradation detected in {metric_name}',
            timestamp=datetime.now(),
            metrics={metric_name: stats},
            recommendations=[
                'Review recent configuration changes',
                'Check for resource constraints',
                'Consider performance optimization'
            ]
        )
        
        with self._lock:
            self.alerts.append(alert)
        self.logger.warning(f"Performance trend alert: {alert.message}")
    
    def _check_system_resources(self) -> None:
        """Check system resource usage."""
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
            
            if memory_usage_mb > self.thresholds['max_memory_usage_mb']:
                self._create_resource_alert('memory', memory_usage_mb)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.thresholds['max_cpu_usage_percent']:
                self._create_resource_alert('cpu', cpu_percent)
                
        except ImportError:
            self.logger.debug("psutil not available for system monitoring")
        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")
    
    def _create_resource_alert(self, resource_type: str, usage: float) -> None:
        """Create alert for resource usage."""
        alert = PerformanceAlert(
            alert_type=f'high_{resource_type}_usage',
            severity='high',
            message=f'High {resource_type} usage: {usage:.1f}%',
            timestamp=datetime.now(),
            metrics={f'{resource_type}_usage': usage},
            recommendations=[
                'Consider reducing cache sizes',
                'Enable more aggressive cleanup',
                'Review batch processing settings'
            ]
        )
        
        with self._lock:
            self.alerts.append(alert)
        self.logger.warning(f"Resource alert: {alert.message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            summary = {
                'monitoring_status': {
                    'active': self.monitoring_active,
                    'metrics_tracked': len(self.metrics_history),
                    'alerts_count': len(self.alerts)
                },
                'recent_metrics': self._get_recent_metrics_summary(),
                'performance_issues': self._get_performance_issues(),
                'recommendations': self._get_optimization_recommendations()
            }
        
        return summary
    
    def _get_recent_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        summary = {}
        
        for metric_name, history in self.metrics_history.items():
            if len(history) > 0:
                recent_values = [entry['value'] for entry in list(history)[-10:]]
                summary[metric_name] = {
                    'current': recent_values[-1] if recent_values else 0,
                    'average': sum(recent_values) / len(recent_values),
                    'min': min(recent_values),
                    'max': max(recent_values),
                    'trend': self._calculate_trend(recent_values)
                }
        
        return summary
    
    def _get_performance_issues(self) -> List[Dict[str, Any]]:
        """Get current performance issues."""
        issues = []
        
        with self._lock:
            for alert in list(self.alerts)[-10:]:  # Last 10 alerts
                issues.append({
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'recommendations': alert.recommendations
                })
        
        return issues
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current metrics."""
        recommendations = []
        
        with self._lock:
            # Analyze cache performance
            if 'cache_hit_rate' in self.metrics_history:
                recent_hit_rates = [entry['value'] for entry in list(self.metrics_history['cache_hit_rate'])[-5:]]
                avg_hit_rate = sum(recent_hit_rates) / len(recent_hit_rates)
                
                if avg_hit_rate < 0.8:
                    recommendations.append("Consider increasing cache sizes or optimizing cache content")
            
            # Analyze processing time
            if 'processing_time_ms' in self.metrics_history:
                recent_times = [entry['value'] for entry in list(self.metrics_history['processing_time_ms'])[-5:]]
                avg_time = sum(recent_times) / len(recent_times)
                
                if avg_time > 50:  # 50ms threshold
                    recommendations.append("Enable batch processing for large texts")
                    recommendations.append("Review confidence thresholds for early termination")
            
            # Analyze unknown word rate
            if 'unknown_word_rate' in self.metrics_history:
                recent_rates = [entry['value'] for entry in list(self.metrics_history['unknown_word_rate'])[-5:]]
                avg_rate = sum(recent_rates) / len(recent_rates)
                
                if avg_rate > 0.2:  # 20% unknown words
                    recommendations.append("Consider expanding vocabulary or improving inference")
        
        return recommendations
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_alerts = [
                {
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'metrics': alert.metrics,
                    'recommendations': alert.recommendations
                }
                for alert in self.alerts
                if alert.timestamp >= cutoff_time
            ]
        
        return recent_alerts
    
    def clear_metrics(self) -> None:
        """Clear all performance metrics."""
        with self._lock:
            self.metrics_history.clear()
            self.alerts.clear()
        self.logger.info("Performance metrics cleared")
    
    def export_metrics(self, file_path: str) -> None:
        """Export performance metrics to file."""
        import json
        
        with self._lock:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'metrics_history': {
                    name: list(history) for name, history in self.metrics_history.items()
                },
                'alerts': [
                    {
                        'type': alert.alert_type,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'metrics': alert.metrics,
                        'recommendations': alert.recommendations
                    }
                    for alert in self.alerts
                ],
                'summary': self.get_performance_summary()
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Performance metrics exported to {file_path}")


# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def start_performance_monitoring(interval_seconds: int = 30) -> None:
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring(interval_seconds)

def stop_performance_monitoring() -> None:
    """Stop global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()
