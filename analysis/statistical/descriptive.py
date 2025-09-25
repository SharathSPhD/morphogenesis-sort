"""Descriptive statistics and distribution analysis for simulation data.

Provides comprehensive descriptive statistical analysis including:
- Basic descriptive statistics (mean, median, variance, etc.)
- Distribution fitting and goodness-of-fit testing
- Outlier detection and analysis
- Data quality assessment
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings
from collections import defaultdict
import logging


@dataclass
class DescriptiveStats:
    """Container for descriptive statistics."""
    count: int
    mean: float
    median: float
    mode: Optional[float]
    std: float
    variance: float
    skewness: float
    kurtosis: float
    minimum: float
    maximum: float
    range: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    iqr: float  # Interquartile range
    mad: float  # Median absolute deviation
    cv: float   # Coefficient of variation
    sem: float  # Standard error of mean


@dataclass
class DistributionFit:
    """Results of distribution fitting."""
    distribution_name: str
    parameters: Tuple[float, ...]
    log_likelihood: float
    aic: float
    bic: float
    ks_statistic: float
    ks_p_value: float
    goodness_of_fit: float
    parameter_errors: Optional[Tuple[float, ...]] = None


@dataclass
class OutlierAnalysis:
    """Results of outlier analysis."""
    method: str
    outlier_indices: List[int]
    outlier_values: List[float]
    threshold_lower: Optional[float] = None
    threshold_upper: Optional[float] = None
    outlier_scores: Optional[List[float]] = None


class DescriptiveStatistics:
    """Comprehensive descriptive statistics analyzer."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compute_descriptive_stats(self, data: Union[List, np.ndarray, pd.Series]) -> DescriptiveStats:
        """Compute comprehensive descriptive statistics.

        Args:
            data: Input data array

        Returns:
            DescriptiveStats object with computed statistics
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values

        # Remove NaN values
        clean_data = data[~np.isnan(data)]

        if len(clean_data) == 0:
            raise ValueError("No valid data points after removing NaN values")

        # Basic statistics
        count = len(clean_data)
        mean = np.mean(clean_data)
        median = np.median(clean_data)
        std = np.std(clean_data, ddof=1) if count > 1 else 0.0
        variance = np.var(clean_data, ddof=1) if count > 1 else 0.0

        # Percentiles
        q25 = np.percentile(clean_data, 25)
        q75 = np.percentile(clean_data, 75)
        iqr = q75 - q25

        # Range statistics
        minimum = np.min(clean_data)
        maximum = np.max(clean_data)
        data_range = maximum - minimum

        # Shape statistics
        skewness = stats.skew(clean_data) if count > 2 else 0.0
        kurtosis = stats.kurtosis(clean_data) if count > 3 else 0.0

        # Other statistics
        mad = np.median(np.abs(clean_data - median))  # Median absolute deviation
        cv = (std / mean) if mean != 0 else float('inf')  # Coefficient of variation
        sem = std / np.sqrt(count) if count > 0 else 0.0  # Standard error of mean

        # Mode calculation
        try:
            mode_result = stats.mode(clean_data, keepdims=True)
            mode_value = mode_result.mode[0] if len(mode_result.mode) > 0 else None
        except Exception:
            mode_value = None

        return DescriptiveStats(
            count=count,
            mean=mean,
            median=median,
            mode=mode_value,
            std=std,
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            minimum=minimum,
            maximum=maximum,
            range=data_range,
            q25=q25,
            q75=q75,
            iqr=iqr,
            mad=mad,
            cv=cv,
            sem=sem
        )

    def compute_grouped_statistics(
        self,
        data: pd.DataFrame,
        group_column: str,
        value_column: str
    ) -> Dict[str, DescriptiveStats]:
        """Compute descriptive statistics for groups.

        Args:
            data: Input DataFrame
            group_column: Column name for grouping
            value_column: Column name for values

        Returns:
            Dictionary mapping group names to DescriptiveStats
        """
        grouped_stats = {}

        for group_name, group_data in data.groupby(group_column):
            values = group_data[value_column].dropna().values
            if len(values) > 0:
                grouped_stats[str(group_name)] = self.compute_descriptive_stats(values)

        return grouped_stats

    def detect_outliers_iqr(
        self,
        data: Union[List, np.ndarray, pd.Series],
        multiplier: float = 1.5
    ) -> OutlierAnalysis:
        """Detect outliers using IQR method.

        Args:
            data: Input data
            multiplier: IQR multiplier for outlier detection

        Returns:
            OutlierAnalysis results
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values

        clean_data = data[~np.isnan(data)]

        q25 = np.percentile(clean_data, 25)
        q75 = np.percentile(clean_data, 75)
        iqr = q75 - q25

        lower_threshold = q25 - multiplier * iqr
        upper_threshold = q75 + multiplier * iqr

        outlier_mask = (clean_data < lower_threshold) | (clean_data > upper_threshold)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = clean_data[outlier_mask].tolist()

        return OutlierAnalysis(
            method="iqr",
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            threshold_lower=lower_threshold,
            threshold_upper=upper_threshold
        )

    def detect_outliers_zscore(
        self,
        data: Union[List, np.ndarray, pd.Series],
        threshold: float = 3.0
    ) -> OutlierAnalysis:
        """Detect outliers using Z-score method.

        Args:
            data: Input data
            threshold: Z-score threshold for outlier detection

        Returns:
            OutlierAnalysis results
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values

        clean_data = data[~np.isnan(data)]

        z_scores = np.abs(stats.zscore(clean_data))
        outlier_mask = z_scores > threshold

        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = clean_data[outlier_mask].tolist()
        outlier_scores = z_scores[outlier_mask].tolist()

        return OutlierAnalysis(
            method="zscore",
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_scores=outlier_scores
        )

    def detect_outliers_isolation_forest(
        self,
        data: Union[List, np.ndarray, pd.Series],
        contamination: float = 0.1
    ) -> OutlierAnalysis:
        """Detect outliers using Isolation Forest.

        Args:
            data: Input data
            contamination: Expected proportion of outliers

        Returns:
            OutlierAnalysis results
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            self.logger.warning("scikit-learn not available, skipping Isolation Forest outlier detection")
            return OutlierAnalysis(method="isolation_forest", outlier_indices=[], outlier_values=[])

        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values

        clean_data = data[~np.isnan(data)].reshape(-1, 1)

        if len(clean_data) < 10:
            self.logger.warning("Insufficient data for Isolation Forest outlier detection")
            return OutlierAnalysis(method="isolation_forest", outlier_indices=[], outlier_values=[])

        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(clean_data)
        outlier_scores = iso_forest.decision_function(clean_data)

        # Extract outliers
        outlier_mask = outlier_labels == -1
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = clean_data[outlier_mask].flatten().tolist()
        outlier_score_values = outlier_scores[outlier_mask].tolist()

        return OutlierAnalysis(
            method="isolation_forest",
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_scores=outlier_score_values
        )

    def assess_data_quality(self, data: Union[pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        """Assess data quality metrics.

        Args:
            data: Input data

        Returns:
            Dictionary with data quality metrics
        """
        if isinstance(data, pd.Series):
            data = pd.DataFrame({'value': data})

        quality_metrics = {}

        for column in data.columns:
            col_data = data[column]

            # Basic counts
            total_count = len(col_data)
            null_count = col_data.isnull().sum()
            valid_count = total_count - null_count
            unique_count = col_data.nunique()

            # Percentages
            null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
            unique_percentage = (unique_count / valid_count) * 100 if valid_count > 0 else 0

            # For numeric data
            quality_info = {
                'total_count': total_count,
                'valid_count': valid_count,
                'null_count': null_count,
                'null_percentage': null_percentage,
                'unique_count': unique_count,
                'unique_percentage': unique_percentage,
                'data_type': str(col_data.dtype)
            }

            # Additional metrics for numeric data
            if pd.api.types.is_numeric_dtype(col_data):
                numeric_data = col_data.dropna()

                if len(numeric_data) > 0:
                    # Infinite values
                    inf_count = np.isinf(numeric_data).sum()
                    quality_info['infinite_count'] = inf_count
                    quality_info['infinite_percentage'] = (inf_count / len(numeric_data)) * 100

                    # Outliers (using IQR method)
                    outlier_analysis = self.detect_outliers_iqr(numeric_data)
                    quality_info['outlier_count'] = len(outlier_analysis.outlier_indices)
                    quality_info['outlier_percentage'] = (
                        len(outlier_analysis.outlier_indices) / len(numeric_data)
                    ) * 100

                    # Zero values
                    zero_count = (numeric_data == 0).sum()
                    quality_info['zero_count'] = zero_count
                    quality_info['zero_percentage'] = (zero_count / len(numeric_data)) * 100

                    # Negative values
                    negative_count = (numeric_data < 0).sum()
                    quality_info['negative_count'] = negative_count
                    quality_info['negative_percentage'] = (negative_count / len(numeric_data)) * 100

            quality_metrics[column] = quality_info

        return quality_metrics


class DistributionAnalysis:
    """Distribution fitting and analysis."""

    # Common distributions to test
    DISTRIBUTIONS = [
        stats.norm,       # Normal
        stats.expon,      # Exponential
        stats.gamma,      # Gamma
        stats.beta,       # Beta
        stats.lognorm,    # Log-normal
        stats.weibull_min, # Weibull
        stats.uniform,    # Uniform
        stats.pareto,     # Pareto
    ]

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fit_distribution(
        self,
        data: Union[List, np.ndarray, pd.Series],
        distribution: stats.rv_continuous
    ) -> DistributionFit:
        """Fit a specific distribution to data.

        Args:
            data: Input data
            distribution: scipy.stats distribution object

        Returns:
            DistributionFit results
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values

        clean_data = data[~np.isnan(data)]

        if len(clean_data) < 5:
            raise ValueError("Insufficient data points for distribution fitting")

        try:
            # Fit distribution
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = distribution.fit(clean_data)

            # Calculate log-likelihood
            log_likelihood = np.sum(distribution.logpdf(clean_data, *params))

            # Calculate AIC and BIC
            k = len(params)  # Number of parameters
            n = len(clean_data)  # Number of data points
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.kstest(clean_data, distribution.cdf, args=params)

            # Goodness of fit score (1 - KS statistic)
            goodness_of_fit = 1 - ks_statistic

            return DistributionFit(
                distribution_name=distribution.name,
                parameters=params,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                ks_statistic=ks_statistic,
                ks_p_value=ks_p_value,
                goodness_of_fit=goodness_of_fit
            )

        except Exception as e:
            self.logger.error(f"Failed to fit {distribution.name} distribution: {e}")
            raise

    def fit_best_distribution(
        self,
        data: Union[List, np.ndarray, pd.Series],
        distributions: Optional[List[stats.rv_continuous]] = None
    ) -> Tuple[DistributionFit, List[DistributionFit]]:
        """Find the best-fitting distribution from a list of candidates.

        Args:
            data: Input data
            distributions: List of distributions to test (defaults to common distributions)

        Returns:
            Tuple of (best_fit, all_fits) where best_fit is the best distribution
            and all_fits is a list of all fitted distributions sorted by goodness of fit
        """
        if distributions is None:
            distributions = self.DISTRIBUTIONS

        fits = []

        for dist in distributions:
            try:
                fit = self.fit_distribution(data, dist)
                fits.append(fit)
            except Exception as e:
                self.logger.debug(f"Failed to fit {dist.name}: {e}")
                continue

        if not fits:
            raise ValueError("Could not fit any distributions to the data")

        # Sort by goodness of fit (descending)
        fits.sort(key=lambda x: x.goodness_of_fit, reverse=True)

        return fits[0], fits

    def compare_distributions(
        self,
        data: Union[List, np.ndarray, pd.Series],
        distributions: List[stats.rv_continuous],
        criterion: str = "aic"
    ) -> List[DistributionFit]:
        """Compare multiple distributions using an information criterion.

        Args:
            data: Input data
            distributions: List of distributions to compare
            criterion: Information criterion ("aic", "bic", or "goodness_of_fit")

        Returns:
            List of DistributionFit objects sorted by the criterion
        """
        fits = []

        for dist in distributions:
            try:
                fit = self.fit_distribution(data, dist)
                fits.append(fit)
            except Exception as e:
                self.logger.debug(f"Failed to fit {dist.name}: {e}")
                continue

        # Sort by criterion
        if criterion == "aic":
            fits.sort(key=lambda x: x.aic)  # Lower is better
        elif criterion == "bic":
            fits.sort(key=lambda x: x.bic)  # Lower is better
        elif criterion == "goodness_of_fit":
            fits.sort(key=lambda x: x.goodness_of_fit, reverse=True)  # Higher is better
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        return fits

    def test_normality(
        self,
        data: Union[List, np.ndarray, pd.Series]
    ) -> Dict[str, Dict[str, float]]:
        """Test for normality using multiple tests.

        Args:
            data: Input data

        Returns:
            Dictionary with results from different normality tests
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values

        clean_data = data[~np.isnan(data)]

        if len(clean_data) < 8:
            raise ValueError("Insufficient data points for normality testing")

        results = {}

        # Shapiro-Wilk test
        try:
            sw_stat, sw_p = stats.shapiro(clean_data)
            results['shapiro_wilk'] = {
                'statistic': sw_stat,
                'p_value': sw_p,
                'is_normal': sw_p > 0.05
            }
        except Exception as e:
            self.logger.debug(f"Shapiro-Wilk test failed: {e}")

        # Anderson-Darling test
        try:
            ad_result = stats.anderson(clean_data, dist='norm')
            # Use 5% significance level (index 2)
            is_normal = ad_result.statistic < ad_result.critical_values[2]
            results['anderson_darling'] = {
                'statistic': ad_result.statistic,
                'critical_values': ad_result.critical_values.tolist(),
                'significance_levels': ad_result.significance_level.tolist(),
                'is_normal': is_normal
            }
        except Exception as e:
            self.logger.debug(f"Anderson-Darling test failed: {e}")

        # D'Agostino's test
        try:
            da_stat, da_p = stats.normaltest(clean_data)
            results['dagostino'] = {
                'statistic': da_stat,
                'p_value': da_p,
                'is_normal': da_p > 0.05
            }
        except Exception as e:
            self.logger.debug(f"D'Agostino test failed: {e}")

        return results

    def generate_qq_plot_data(
        self,
        data: Union[List, np.ndarray, pd.Series],
        distribution: stats.rv_continuous = stats.norm
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data for Q-Q plot.

        Args:
            data: Input data
            distribution: Reference distribution for Q-Q plot

        Returns:
            Tuple of (theoretical_quantiles, sample_quantiles)
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values

        clean_data = np.sort(data[~np.isnan(data)])
        n = len(clean_data)

        # Calculate sample quantiles (empirical)
        sample_quantiles = clean_data

        # Calculate theoretical quantiles
        probabilities = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = distribution.ppf(probabilities)

        return theoretical_quantiles, sample_quantiles