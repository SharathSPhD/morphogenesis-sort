"""Hypothesis testing and significance analysis for morphogenesis simulations.

Provides comprehensive statistical hypothesis testing capabilities for analyzing
simulation results, comparing experimental conditions, and determining statistical
significance of observed effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path

try:
    from scipy import stats
    from scipy.stats import (
        normaltest, kstest, shapiro, anderson,
        ttest_1samp, ttest_ind, ttest_rel,
        mannwhitneyu, wilcoxon, kruskal,
        chi2_contingency, fisher_exact,
        pearsonr, spearmanr, kendalltau,
        f_oneway, bartlett, levene
    )
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some statistical tests will be limited.")


class TestType(Enum):
    """Statistical test types."""
    NORMALITY = "normality"
    ONE_SAMPLE = "one_sample"
    TWO_SAMPLE = "two_sample"
    PAIRED = "paired"
    NON_PARAMETRIC = "non_parametric"
    VARIANCE = "variance"
    CORRELATION = "correlation"
    CONTINGENCY = "contingency"
    MULTIPLE_COMPARISON = "multiple_comparison"


class AlternativeHypothesis(Enum):
    """Alternative hypothesis types."""
    TWO_SIDED = "two-sided"
    LESS = "less"
    GREATER = "greater"


@dataclass
class TestResult:
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    degrees_of_freedom: Optional[int] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    method: str = ""
    sample_size: int = 0
    alpha: float = 0.05
    power: Optional[float] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        """Check if result is statistically significant."""
        return self.p_value < self.alpha

    def summary(self) -> str:
        """Generate summary text."""
        significance = "significant" if self.is_significant else "not significant"
        return (
            f"{self.test_name}: statistic={self.statistic:.4f}, "
            f"p-value={self.p_value:.4f}, {significance} at α={self.alpha}"
        )


@dataclass
class MultipleComparisonResult:
    """Multiple comparison test result."""
    method: str
    comparisons: List[Tuple[str, str]]  # Group pairs
    statistics: List[float]
    p_values: List[float]
    adjusted_p_values: List[float]
    significant: List[bool]
    alpha: float = 0.05
    correction_method: str = "bonferroni"


class HypothesisTests:
    """Comprehensive hypothesis testing framework."""

    def __init__(self, alpha: float = 0.05, random_state: Optional[int] = None):
        """Initialize hypothesis testing framework.

        Args:
            alpha: Significance level
            random_state: Random seed for reproducibility
        """
        self.alpha = alpha
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def test_normality(
        self,
        data: Union[np.ndarray, List[float], pd.Series],
        method: str = "shapiro"
    ) -> TestResult:
        """Test for normality of data distribution.

        Args:
            data: Sample data
            method: Test method ('shapiro', 'normaltest', 'ks', 'anderson')

        Returns:
            TestResult with normality test results
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]  # Remove NaN values

        if not SCIPY_AVAILABLE:
            return self._fallback_normality_test(data)

        if method == "shapiro":
            if len(data) > 5000:
                warnings.warn("Shapiro-Wilk test may be unreliable for large samples")
            stat, p_value = shapiro(data)
            test_name = "Shapiro-Wilk Test"

        elif method == "normaltest":
            stat, p_value = normaltest(data)
            test_name = "D'Agostino and Pearson Test"

        elif method == "ks":
            # Kolmogorov-Smirnov test against normal distribution
            mean, std = data.mean(), data.std()
            stat, p_value = kstest(data, lambda x: stats.norm.cdf(x, mean, std))
            test_name = "Kolmogorov-Smirnov Test"

        elif method == "anderson":
            result = anderson(data, dist='norm')
            stat = result.statistic
            # Use 5% significance level
            critical_value = result.critical_values[2]
            p_value = 0.05 if stat > critical_value else 0.1
            test_name = "Anderson-Darling Test"

        else:
            raise ValueError(f"Unknown normality test method: {method}")

        interpretation = (
            "Data appears normally distributed" if p_value >= self.alpha
            else "Data does not appear normally distributed"
        )

        return TestResult(
            test_name=test_name,
            statistic=stat,
            p_value=p_value,
            interpretation=interpretation,
            method=method,
            sample_size=len(data),
            alpha=self.alpha
        )

    def one_sample_t_test(
        self,
        data: Union[np.ndarray, List[float], pd.Series],
        population_mean: float,
        alternative: str = "two-sided"
    ) -> TestResult:
        """One-sample t-test.

        Args:
            data: Sample data
            population_mean: Hypothesized population mean
            alternative: Alternative hypothesis

        Returns:
            TestResult with t-test results
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]

        if not SCIPY_AVAILABLE:
            return self._fallback_t_test(data, population_mean, alternative)

        stat, p_value = ttest_1samp(data, population_mean, alternative=alternative)

        # Calculate effect size (Cohen's d)
        effect_size = (data.mean() - population_mean) / data.std(ddof=1)

        # Confidence interval
        se = stats.sem(data)
        df = len(data) - 1
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        margin = t_critical * se
        ci = (data.mean() - margin, data.mean() + margin)

        interpretation = (
            f"Sample mean ({data.mean():.4f}) "
            f"{'differs significantly from' if p_value < self.alpha else 'does not differ significantly from'} "
            f"population mean ({population_mean})"
        )

        return TestResult(
            test_name="One-Sample t-Test",
            statistic=stat,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation,
            method=f"t-test (alternative: {alternative})",
            sample_size=len(data),
            alpha=self.alpha
        )

    def independent_samples_t_test(
        self,
        group1: Union[np.ndarray, List[float], pd.Series],
        group2: Union[np.ndarray, List[float], pd.Series],
        equal_variances: bool = True,
        alternative: str = "two-sided"
    ) -> TestResult:
        """Independent samples t-test.

        Args:
            group1: First group data
            group2: Second group data
            equal_variances: Assume equal variances (Welch's t-test if False)
            alternative: Alternative hypothesis

        Returns:
            TestResult with t-test results
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]

        if not SCIPY_AVAILABLE:
            return self._fallback_independent_t_test(group1, group2, equal_variances)

        stat, p_value = ttest_ind(
            group1, group2,
            equal_var=equal_variances,
            alternative=alternative
        )

        # Calculate pooled effect size (Cohen's d)
        if equal_variances:
            pooled_std = np.sqrt(
                ((len(group1) - 1) * group1.var(ddof=1) +
                 (len(group2) - 1) * group2.var(ddof=1)) /
                (len(group1) + len(group2) - 2)
            )
            effect_size = (group1.mean() - group2.mean()) / pooled_std
        else:
            # Glass's delta for unequal variances
            effect_size = (group1.mean() - group2.mean()) / group2.std(ddof=1)

        method = f"{'Welch\'s' if not equal_variances else 'Student\'s'} t-test"
        interpretation = (
            f"Groups {'differ significantly' if p_value < self.alpha else 'do not differ significantly'} "
            f"(Group1 mean: {group1.mean():.4f}, Group2 mean: {group2.mean():.4f})"
        )

        return TestResult(
            test_name="Independent Samples t-Test",
            statistic=stat,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation,
            method=method,
            sample_size=len(group1) + len(group2),
            alpha=self.alpha,
            additional_info={
                "group1_size": len(group1),
                "group2_size": len(group2),
                "equal_variances": equal_variances
            }
        )

    def paired_samples_t_test(
        self,
        pre: Union[np.ndarray, List[float], pd.Series],
        post: Union[np.ndarray, List[float], pd.Series],
        alternative: str = "two-sided"
    ) -> TestResult:
        """Paired samples t-test.

        Args:
            pre: Pre-treatment measurements
            post: Post-treatment measurements
            alternative: Alternative hypothesis

        Returns:
            TestResult with paired t-test results
        """
        pre = np.asarray(pre)
        post = np.asarray(post)

        if len(pre) != len(post):
            raise ValueError("Pre and post samples must have equal length")

        # Remove pairs with NaN values
        valid_mask = ~(np.isnan(pre) | np.isnan(post))
        pre = pre[valid_mask]
        post = post[valid_mask]

        if not SCIPY_AVAILABLE:
            return self._fallback_paired_t_test(pre, post, alternative)

        stat, p_value = ttest_rel(pre, post, alternative=alternative)

        # Calculate effect size (Cohen's d for paired samples)
        differences = post - pre
        effect_size = differences.mean() / differences.std(ddof=1)

        interpretation = (
            f"{'Significant change' if p_value < self.alpha else 'No significant change'} "
            f"from pre ({pre.mean():.4f}) to post ({post.mean():.4f}) measurements"
        )

        return TestResult(
            test_name="Paired Samples t-Test",
            statistic=stat,
            p_value=p_value,
            degrees_of_freedom=len(pre) - 1,
            effect_size=effect_size,
            interpretation=interpretation,
            method=f"paired t-test (alternative: {alternative})",
            sample_size=len(pre),
            alpha=self.alpha,
            additional_info={"mean_difference": differences.mean()}
        )

    def mann_whitney_u_test(
        self,
        group1: Union[np.ndarray, List[float], pd.Series],
        group2: Union[np.ndarray, List[float], pd.Series],
        alternative: str = "two-sided"
    ) -> TestResult:
        """Mann-Whitney U test (non-parametric alternative to independent t-test).

        Args:
            group1: First group data
            group2: Second group data
            alternative: Alternative hypothesis

        Returns:
            TestResult with Mann-Whitney U test results
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]

        if not SCIPY_AVAILABLE:
            return self._fallback_mann_whitney(group1, group2)

        stat, p_value = mannwhitneyu(
            group1, group2,
            alternative=alternative
        )

        # Effect size (rank biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * stat) / (n1 * n2)

        interpretation = (
            f"Groups {'have significantly different distributions' if p_value < self.alpha else 'do not have significantly different distributions'}"
        )

        return TestResult(
            test_name="Mann-Whitney U Test",
            statistic=stat,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation,
            method=f"Mann-Whitney U (alternative: {alternative})",
            sample_size=n1 + n2,
            alpha=self.alpha,
            additional_info={
                "group1_size": n1,
                "group2_size": n2
            }
        )

    def correlation_test(
        self,
        x: Union[np.ndarray, List[float], pd.Series],
        y: Union[np.ndarray, List[float], pd.Series],
        method: str = "pearson"
    ) -> TestResult:
        """Test correlation between two variables.

        Args:
            x: First variable
            y: Second variable
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            TestResult with correlation test results
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if len(x) != len(y):
            raise ValueError("Variables must have equal length")

        # Remove pairs with NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]

        if not SCIPY_AVAILABLE:
            return self._fallback_correlation(x, y, method)

        if method == "pearson":
            stat, p_value = pearsonr(x, y)
            test_name = "Pearson Correlation Test"
        elif method == "spearman":
            stat, p_value = spearmanr(x, y)
            test_name = "Spearman Rank Correlation Test"
        elif method == "kendall":
            stat, p_value = kendalltau(x, y)
            test_name = "Kendall Tau Correlation Test"
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        # Effect size is the correlation coefficient itself
        effect_size = abs(stat)

        strength = self._interpret_correlation_strength(effect_size)
        interpretation = (
            f"{'Significant' if p_value < self.alpha else 'Non-significant'} "
            f"{strength} {method} correlation (r = {stat:.4f})"
        )

        return TestResult(
            test_name=test_name,
            statistic=stat,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation,
            method=method,
            sample_size=len(x),
            alpha=self.alpha
        )

    def _interpret_correlation_strength(self, r: float) -> str:
        """Interpret correlation strength."""
        r = abs(r)
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "weak"
        elif r < 0.5:
            return "moderate"
        elif r < 0.7:
            return "strong"
        else:
            return "very strong"

    def _fallback_normality_test(self, data: np.ndarray) -> TestResult:
        """Fallback normality test using basic statistics."""
        # Simple normality check using skewness and kurtosis
        n = len(data)
        mean = data.mean()
        std = data.std(ddof=1)

        # Standardize data
        z_scores = (data - mean) / std

        # Calculate skewness and kurtosis
        skewness = np.mean(z_scores**3)
        kurtosis = np.mean(z_scores**4) - 3

        # Simple test statistic
        stat = abs(skewness) + abs(kurtosis)

        # Rough p-value approximation
        p_value = max(0.01, min(0.99, 1 - stat/2))

        return TestResult(
            test_name="Basic Normality Check",
            statistic=stat,
            p_value=p_value,
            interpretation="Approximate normality assessment (SciPy not available)",
            method="skewness+kurtosis",
            sample_size=n,
            alpha=self.alpha
        )

    def _fallback_t_test(self, data: np.ndarray, pop_mean: float, alternative: str) -> TestResult:
        """Fallback t-test implementation."""
        n = len(data)
        sample_mean = data.mean()
        sample_std = data.std(ddof=1)
        se = sample_std / np.sqrt(n)

        t_stat = (sample_mean - pop_mean) / se

        # Rough p-value approximation
        p_value = max(0.001, min(0.999, 2 * (1 - abs(t_stat) / (abs(t_stat) + 2))))

        return TestResult(
            test_name="Approximate t-Test",
            statistic=t_stat,
            p_value=p_value,
            interpretation="Approximate t-test (SciPy not available)",
            method="approximate",
            sample_size=n,
            alpha=self.alpha
        )

    def _fallback_independent_t_test(self, group1: np.ndarray, group2: np.ndarray, equal_var: bool) -> TestResult:
        """Fallback independent t-test."""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = group1.mean(), group2.mean()
        var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

        if equal_var:
            pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
        else:
            se = np.sqrt(var1/n1 + var2/n2)

        t_stat = (mean1 - mean2) / se
        p_value = max(0.001, min(0.999, 2 * (1 - abs(t_stat) / (abs(t_stat) + 2))))

        return TestResult(
            test_name="Approximate Independent t-Test",
            statistic=t_stat,
            p_value=p_value,
            interpretation="Approximate independent t-test (SciPy not available)",
            method="approximate",
            sample_size=n1 + n2,
            alpha=self.alpha
        )

    def _fallback_paired_t_test(self, pre: np.ndarray, post: np.ndarray, alternative: str) -> TestResult:
        """Fallback paired t-test."""
        differences = post - pre
        return self._fallback_t_test(differences, 0.0, alternative)

    def _fallback_mann_whitney(self, group1: np.ndarray, group2: np.ndarray) -> TestResult:
        """Fallback Mann-Whitney test."""
        # Simple rank-based approximation
        combined = np.concatenate([group1, group2])
        ranks = stats.rankdata(combined) if SCIPY_AVAILABLE else np.argsort(np.argsort(combined)) + 1

        group1_ranks = ranks[:len(group1)]
        u_stat = np.sum(group1_ranks) - len(group1) * (len(group1) + 1) / 2

        # Rough p-value
        expected = len(group1) * len(group2) / 2
        p_value = max(0.001, min(0.999, 1 - abs(u_stat - expected) / (expected + 1)))

        return TestResult(
            test_name="Approximate Mann-Whitney U Test",
            statistic=u_stat,
            p_value=p_value,
            interpretation="Approximate Mann-Whitney test (SciPy not available)",
            method="approximate",
            sample_size=len(group1) + len(group2),
            alpha=self.alpha
        )

    def _fallback_correlation(self, x: np.ndarray, y: np.ndarray, method: str) -> TestResult:
        """Fallback correlation test."""
        if method == "pearson" or not SCIPY_AVAILABLE:
            # Basic Pearson correlation
            r = np.corrcoef(x, y)[0, 1]
        else:
            # Use scipy for other methods if available
            if method == "spearman":
                r, _ = spearmanr(x, y)
            elif method == "kendall":
                r, _ = kendalltau(x, y)

        # Rough p-value approximation
        n = len(x)
        t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 0.999 else 100
        p_value = max(0.001, min(0.999, 2 * (1 - abs(t_stat) / (abs(t_stat) + 2))))

        return TestResult(
            test_name=f"Approximate {method.title()} Correlation",
            statistic=r,
            p_value=p_value,
            interpretation=f"Approximate {method} correlation (limited functionality)",
            method="approximate",
            sample_size=n,
            alpha=self.alpha
        )


class SignificanceAnalyzer:
    """Advanced significance analysis and multiple comparisons."""

    def __init__(self, alpha: float = 0.05):
        """Initialize significance analyzer.

        Args:
            alpha: Significance level
        """
        self.alpha = alpha

    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: Optional[float] = None
    ) -> Tuple[List[float], List[bool]]:
        """Apply Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values
            alpha: Significance level (uses instance alpha if None)

        Returns:
            Tuple of (adjusted_p_values, significant_flags)
        """
        if alpha is None:
            alpha = self.alpha

        n_tests = len(p_values)
        adjusted_alpha = alpha / n_tests

        adjusted_p_values = [min(1.0, p * n_tests) for p in p_values]
        significant = [p < adjusted_alpha for p in p_values]

        return adjusted_p_values, significant

    def false_discovery_rate(
        self,
        p_values: List[float],
        alpha: Optional[float] = None,
        method: str = "bh"
    ) -> Tuple[List[float], List[bool]]:
        """Apply False Discovery Rate correction.

        Args:
            p_values: List of p-values
            alpha: Significance level
            method: Correction method ('bh' for Benjamini-Hochberg)

        Returns:
            Tuple of (adjusted_p_values, significant_flags)
        """
        if alpha is None:
            alpha = self.alpha

        if method != "bh":
            raise ValueError("Only Benjamini-Hochberg method currently supported")

        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]

        # Benjamini-Hochberg procedure
        adjusted_p_values = np.zeros_like(sorted_p_values)
        for i in range(n_tests - 1, -1, -1):
            if i == n_tests - 1:
                adjusted_p_values[i] = sorted_p_values[i]
            else:
                adjusted_p_values[i] = min(
                    adjusted_p_values[i + 1],
                    sorted_p_values[i] * n_tests / (i + 1)
                )

        # Restore original order
        original_order_adjusted = np.zeros_like(adjusted_p_values)
        original_order_adjusted[sorted_indices] = adjusted_p_values

        significant = original_order_adjusted < alpha

        return original_order_adjusted.tolist(), significant.tolist()

    def effect_size_interpretation(self, effect_size: float, test_type: str) -> str:
        """Interpret effect size magnitude.

        Args:
            effect_size: Effect size value
            test_type: Type of effect size ('cohens_d', 'correlation', 'eta_squared')

        Returns:
            Interpretation string
        """
        effect_size = abs(effect_size)

        if test_type == "cohens_d":
            if effect_size < 0.2:
                return "negligible"
            elif effect_size < 0.5:
                return "small"
            elif effect_size < 0.8:
                return "medium"
            else:
                return "large"

        elif test_type == "correlation":
            if effect_size < 0.1:
                return "negligible"
            elif effect_size < 0.3:
                return "small"
            elif effect_size < 0.5:
                return "medium"
            else:
                return "large"

        elif test_type == "eta_squared":
            if effect_size < 0.01:
                return "negligible"
            elif effect_size < 0.06:
                return "small"
            elif effect_size < 0.14:
                return "medium"
            else:
                return "large"

        else:
            return "unknown"

    def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float,
        test_type: str = "t_test"
    ) -> float:
        """Estimate statistical power (simplified approximation).

        Args:
            effect_size: Expected effect size
            sample_size: Sample size
            alpha: Significance level
            test_type: Type of test

        Returns:
            Estimated power
        """
        # Simplified power calculation
        # In practice, would use more sophisticated methods

        if test_type == "t_test":
            # Approximate power for t-test
            noncentrality = effect_size * np.sqrt(sample_size / 2)
        else:
            noncentrality = effect_size * np.sqrt(sample_size)

        # Rough approximation of power
        power = 1 - stats.norm.cdf(
            stats.norm.ppf(1 - alpha/2) - noncentrality
        ) if SCIPY_AVAILABLE else min(0.99, max(0.01, noncentrality / 3))

        return max(0.0, min(1.0, power))


# Example usage and testing
if __name__ == "__main__":
    # Demonstrate hypothesis testing capabilities
    np.random.seed(42)

    # Create test data
    normal_data = np.random.normal(50, 10, 100)
    non_normal_data = np.random.exponential(2, 100)
    group1 = np.random.normal(50, 10, 50)
    group2 = np.random.normal(55, 10, 50)

    # Initialize hypothesis tests
    tests = HypothesisTests(alpha=0.05)

    # Test normality
    print("=== Normality Tests ===")
    norm_result = tests.test_normality(normal_data, method="shapiro")
    print(norm_result.summary())

    non_norm_result = tests.test_normality(non_normal_data, method="shapiro")
    print(non_norm_result.summary())

    # One-sample t-test
    print("\n=== One-Sample t-Test ===")
    one_sample_result = tests.one_sample_t_test(normal_data, population_mean=45)
    print(one_sample_result.summary())
    print(f"Effect size (Cohen's d): {one_sample_result.effect_size:.3f}")

    # Independent samples t-test
    print("\n=== Independent Samples t-Test ===")
    independent_result = tests.independent_samples_t_test(group1, group2)
    print(independent_result.summary())
    print(f"Effect size: {independent_result.effect_size:.3f}")

    # Mann-Whitney U test
    print("\n=== Mann-Whitney U Test ===")
    mw_result = tests.mann_whitney_u_test(group1, group2)
    print(mw_result.summary())

    # Correlation test
    print("\n=== Correlation Tests ===")
    x = np.random.normal(0, 1, 100)
    y = x + np.random.normal(0, 0.5, 100)  # Correlated data

    pearson_result = tests.correlation_test(x, y, method="pearson")
    print(pearson_result.summary())

    # Multiple comparisons
    print("\n=== Multiple Comparisons ===")
    analyzer = SignificanceAnalyzer()
    p_values = [0.01, 0.03, 0.05, 0.07, 0.12]

    bonf_adj, bonf_sig = analyzer.bonferroni_correction(p_values)
    fdr_adj, fdr_sig = analyzer.false_discovery_rate(p_values)

    print(f"Original p-values: {p_values}")
    print(f"Bonferroni adjusted: {[f'{p:.3f}' for p in bonf_adj]}")
    print(f"FDR adjusted: {[f'{p:.3f}' for p in fdr_adj]}")
    print(f"Bonferroni significant: {bonf_sig}")
    print(f"FDR significant: {fdr_sig}")