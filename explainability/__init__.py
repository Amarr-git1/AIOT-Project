"""
Explainability Module - Initialize package

This module provides comprehensive Explainable AI capabilities for the
network security decision-making system.
"""

from explainability.xai_engine import (
    XAIEngine,
    ActionType,
    ActionMetadata,
    FeatureNames,
    FeatureContributionCalculator,
    RuleBasedExplainer,
    NodeActionExplainer,
    GlobalActionExplainer
)

from explainability.xai_integration import (
    ExplainableDecisionWrapper,
    ExplanationReporter,
    ExplanationVisualizer
)

__all__ = [
    'XAIEngine',
    'ActionType',
    'ActionMetadata',
    'FeatureNames',
    'FeatureContributionCalculator',
    'RuleBasedExplainer',
    'NodeActionExplainer',
    'GlobalActionExplainer',
    'ExplainableDecisionWrapper',
    'ExplanationReporter',
    'ExplanationVisualizer'
]

__version__ = '1.0.0'
__author__ = 'Explainable AI Team'
__description__ = 'Comprehensive Explainable AI system for network security decisions'
