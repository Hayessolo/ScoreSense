"""
ScoreSense: A production-grade prediction system with advanced ensemble learning
and statistical validation for high-stakes predictions.
"""

from .models import ScoreSenseEnsemble
from .utils import (
    calculate_metrics,
    calculate_metrics_summary,
    bootstrap_confidence_interval,
    analyze_performance_by_range,
    calculate_performance_based_weights,
    create_features
)

__version__ = '0.1.0'
__author__ = 'Hayes Frank'

__all__ = [
    'ScoreSenseEnsemble',
    'calculate_metrics',
    'calculate_metrics_summary',
    'bootstrap_confidence_interval',
    'analyze_performance_by_range',
    'calculate_performance_based_weights',
    'create_features'
]