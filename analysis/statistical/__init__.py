"""Statistical analysis modules for morphogenesis simulation.

Provides comprehensive statistical analysis capabilities including:
- Descriptive statistics and distributions
- Hypothesis testing and significance analysis
- Time series analysis and trend detection
- Multivariate analysis and dimensionality reduction
- Bayesian inference and model comparison
"""

from .descriptive import DescriptiveStatistics, DistributionAnalysis
from .hypothesis_testing import HypothesisTests, SignificanceAnalyzer
from .time_series import TimeSeriesAnalyzer, TrendAnalysis, SeasonalityDetector
from .multivariate import MultivariateAnalyzer, PCAAnalysis, ClusterAnalysis
from .bayesian import BayesianAnalyzer, ModelComparison, PosteriorAnalysis
from .regression import RegressionAnalyzer, ModelSelection, ResidualAnalysis
from .survival import SurvivalAnalysis, KaplanMeierAnalyzer, CoxAnalysis

__all__ = [
    'DescriptiveStatistics',
    'DistributionAnalysis',
    'HypothesisTests',
    'SignificanceAnalyzer',
    'TimeSeriesAnalyzer',
    'TrendAnalysis',
    'SeasonalityDetector',
    'MultivariateAnalyzer',
    'PCAAnalysis',
    'ClusterAnalysis',
    'BayesianAnalyzer',
    'ModelComparison',
    'PosteriorAnalysis',
    'RegressionAnalyzer',
    'ModelSelection',
    'ResidualAnalysis',
    'SurvivalAnalysis',
    'KaplanMeierAnalyzer',
    'CoxAnalysis',
]