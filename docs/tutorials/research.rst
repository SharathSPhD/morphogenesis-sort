Research Methodology Tutorial
============================

This tutorial covers advanced research methodologies and experimental design for morphogenesis studies, including statistical analysis, hypothesis testing, and publication-ready experimental protocols.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Research in morphogenesis requires rigorous experimental design, proper statistical analysis, and reproducible methodologies. This tutorial provides a comprehensive guide to conducting high-quality morphogenesis research using our platform.

Experimental Design Principles
------------------------------

Hypothesis Formation
~~~~~~~~~~~~~~~~~~~~

Proper hypothesis formation is crucial for meaningful research:

.. code-block:: python

    from morphogenesis.research import HypothesisFramework, ExperimentalDesign
    from morphogenesis.analysis import StatisticalAnalyzer

    class MorphogenesisHypothesis:
        """Framework for formulating testable hypotheses in morphogenesis."""

        def __init__(self, research_question):
            self.research_question = research_question
            self.null_hypothesis = None
            self.alternative_hypothesis = None
            self.variables = {}
            self.predictions = []

        def define_hypotheses(self, null_h, alternative_h):
            """Define null and alternative hypotheses."""
            self.null_hypothesis = null_h
            self.alternative_hypothesis = alternative_h

        def identify_variables(self, independent, dependent, controls):
            """Identify experimental variables."""
            self.variables = {
                'independent': independent,
                'dependent': dependent,
                'controls': controls
            }

        def make_predictions(self, predictions):
            """Define testable predictions."""
            self.predictions = predictions

    # Example: Cell sorting efficiency hypothesis
    hypothesis = MorphogenesisHypothesis(
        "Does morphogen concentration affect cell sorting efficiency?"
    )

    hypothesis.define_hypotheses(
        null_h="Morphogen concentration has no effect on sorting efficiency",
        alternative_h="Higher morphogen concentration increases sorting efficiency"
    )

    hypothesis.identify_variables(
        independent=['morphogen_concentration'],
        dependent=['sorting_efficiency', 'sorting_time'],
        controls=['temperature', 'cell_density', 'simulation_duration']
    )

    hypothesis.make_predictions([
        "Sorting efficiency will increase linearly with morphogen concentration",
        "Sorting time will decrease exponentially with concentration"
    ])

Power Analysis and Sample Size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determine appropriate sample sizes for statistical power:

.. code-block:: python

    import numpy as np
    from scipy import stats
    from morphogenesis.research import PowerAnalysis

    class PowerCalculator:
        """Calculate statistical power and required sample sizes."""

        def __init__(self, effect_size, alpha=0.05, power=0.8):
            self.effect_size = effect_size
            self.alpha = alpha
            self.power = power

        def calculate_sample_size(self, test_type='two_sample'):
            """Calculate minimum sample size for desired power."""
            if test_type == 'two_sample':
                # Cohen's d for two-sample t-test
                z_alpha = stats.norm.ppf(1 - self.alpha/2)
                z_beta = stats.norm.ppf(self.power)

                n = 2 * ((z_alpha + z_beta) / self.effect_size) ** 2
                return int(np.ceil(n))

            elif test_type == 'correlation':
                # Fisher's z-transform for correlation
                z_alpha = stats.norm.ppf(1 - self.alpha/2)
                z_beta = stats.norm.ppf(self.power)

                z_r = 0.5 * np.log((1 + self.effect_size) / (1 - self.effect_size))
                n = ((z_alpha + z_beta) / z_r) ** 2 + 3
                return int(np.ceil(n))

        def calculate_achieved_power(self, n, test_type='two_sample'):
            """Calculate achieved power with given sample size."""
            if test_type == 'two_sample':
                z_alpha = stats.norm.ppf(1 - self.alpha/2)
                z_effect = self.effect_size * np.sqrt(n/2)
                achieved_power = stats.norm.cdf(z_effect - z_alpha)
                return achieved_power

    # Example usage
    power_calc = PowerCalculator(effect_size=0.5, alpha=0.05, power=0.8)
    required_n = power_calc.calculate_sample_size('two_sample')
    print(f"Required sample size per group: {required_n}")

    achieved_power = power_calc.calculate_achieved_power(required_n)
    print(f"Achieved power: {achieved_power:.3f}")

Experimental Controls and Randomization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement proper experimental controls:

.. code-block:: python

    import random
    from morphogenesis.experiments import ControlledExperiment

    class ExperimentalDesign:
        """Design and manage controlled experiments."""

        def __init__(self, factors, levels, replicates=3):
            self.factors = factors
            self.levels = levels
            self.replicates = replicates
            self.design_matrix = None

        def generate_factorial_design(self):
            """Generate full factorial design matrix."""
            import itertools

            factor_combinations = list(itertools.product(*self.levels.values()))

            design = []
            for combination in factor_combinations:
                factor_dict = dict(zip(self.factors, combination))
                for rep in range(self.replicates):
                    design.append({**factor_dict, 'replicate': rep + 1})

            # Randomize order
            random.shuffle(design)
            self.design_matrix = design
            return design

        def generate_blocked_design(self, block_factor):
            """Generate randomized block design."""
            design = self.generate_factorial_design()

            # Group by blocks and randomize within blocks
            blocks = {}
            for trial in design:
                block_value = trial[block_factor]
                if block_value not in blocks:
                    blocks[block_value] = []
                blocks[block_value].append(trial)

            for block in blocks.values():
                random.shuffle(block)

            blocked_design = []
            for block in blocks.values():
                blocked_design.extend(block)

            return blocked_design

    # Example: Multi-factor morphogenesis experiment
    experiment = ExperimentalDesign(
        factors=['morphogen_type', 'concentration', 'cell_type'],
        levels={
            'morphogen_type': ['BMP', 'Wnt', 'FGF'],
            'concentration': [0.1, 0.5, 1.0, 2.0],
            'cell_type': ['epithelial', 'mesenchymal']
        },
        replicates=5
    )

    factorial_design = experiment.generate_factorial_design()
    print(f"Total experimental runs: {len(factorial_design)}")

    # Show first few runs
    for i, run in enumerate(factorial_design[:5]):
        print(f"Run {i+1}: {run}")

Statistical Analysis Methods
---------------------------

Descriptive Statistics
~~~~~~~~~~~~~~~~~~~~~~

Comprehensive descriptive analysis of morphogenesis data:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from scipy import stats
    from morphogenesis.analysis import DescriptiveAnalyzer

    class MorphogenesisDescriptives:
        """Advanced descriptive statistics for morphogenesis research."""

        def __init__(self, data):
            self.data = pd.DataFrame(data)

        def basic_statistics(self, variables=None):
            """Calculate basic descriptive statistics."""
            if variables is None:
                variables = self.data.select_dtypes(include=[np.number]).columns

            results = {}
            for var in variables:
                series = self.data[var].dropna()
                results[var] = {
                    'n': len(series),
                    'mean': series.mean(),
                    'median': series.median(),
                    'mode': series.mode().iloc[0] if not series.mode().empty else None,
                    'std': series.std(),
                    'var': series.var(),
                    'min': series.min(),
                    'max': series.max(),
                    'range': series.max() - series.min(),
                    'q1': series.quantile(0.25),
                    'q3': series.quantile(0.75),
                    'iqr': series.quantile(0.75) - series.quantile(0.25),
                    'skewness': stats.skew(series),
                    'kurtosis': stats.kurtosis(series),
                    'cv': series.std() / series.mean() * 100  # Coefficient of variation
                }

            return results

        def distribution_tests(self, variable):
            """Test for normality and other distribution characteristics."""
            series = self.data[variable].dropna()

            tests = {}

            # Normality tests
            if len(series) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(series)
                tests['shapiro_wilk'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}

            if len(series) >= 8:
                ks_stat, ks_p = stats.kstest(series, 'norm',
                                           args=(series.mean(), series.std()))
                tests['kolmogorov_smirnov'] = {'statistic': ks_stat, 'p_value': ks_p}

            # Outlier detection using IQR method
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]

            tests['outliers'] = {
                'count': len(outliers),
                'values': outliers.tolist(),
                'percentage': len(outliers) / len(series) * 100
            }

            return tests

    # Example usage with morphogenesis simulation data
    simulation_data = {
        'sorting_efficiency': np.random.beta(2, 5, 100) * 100,
        'convergence_time': np.random.gamma(2, 10, 100),
        'cell_displacement': np.random.lognormal(1, 0.5, 100),
        'energy_dissipation': np.random.exponential(2, 100)
    }

    analyzer = MorphogenesisDescriptives(simulation_data)
    descriptives = analyzer.basic_statistics()

    for variable, stats_dict in descriptives.items():
        print(f"\n{variable.upper()}:")
        for stat, value in stats_dict.items():
            if isinstance(value, float):
                print(f"  {stat}: {value:.3f}")
            else:
                print(f"  {stat}: {value}")

Hypothesis Testing
~~~~~~~~~~~~~~~~~~

Comprehensive hypothesis testing framework:

.. code-block:: python

    from scipy import stats
    import numpy as np
    from morphogenesis.analysis import HypothesisTests

    class MorphogenesisTests:
        """Hypothesis testing specifically for morphogenesis research."""

        def __init__(self, alpha=0.05):
            self.alpha = alpha

        def compare_two_groups(self, group1, group2, test_type='auto'):
            """Compare two groups using appropriate statistical test."""
            group1 = np.array(group1)
            group2 = np.array(group2)

            results = {}

            # Check assumptions
            results['assumptions'] = self._check_assumptions(group1, group2)

            if test_type == 'auto':
                if (results['assumptions']['normality_g1'] and
                    results['assumptions']['normality_g2'] and
                    results['assumptions']['equal_variance']):
                    test_type = 'ttest'
                else:
                    test_type = 'mannwhitney'

            if test_type == 'ttest':
                if results['assumptions']['equal_variance']:
                    stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
                    test_name = "Independent t-test (equal variance)"
                else:
                    stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                    test_name = "Welch's t-test (unequal variance)"

                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) +
                                    (len(group2)-1)*np.var(group2, ddof=1)) /
                                   (len(group1)+len(group2)-2))
                cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
                results['effect_size'] = cohens_d

            elif test_type == 'mannwhitney':
                stat, p_value = stats.mannwhitneyu(group1, group2,
                                                 alternative='two-sided')
                test_name = "Mann-Whitney U test"

                # Calculate rank-biserial correlation as effect size
                n1, n2 = len(group1), len(group2)
                u_stat = stat
                effect_size = 1 - (2 * u_stat) / (n1 * n2)
                results['effect_size'] = effect_size

            results.update({
                'test_name': test_name,
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'alpha': self.alpha
            })

            return results

        def _check_assumptions(self, group1, group2):
            """Check statistical test assumptions."""
            assumptions = {}

            # Normality tests
            if len(group1) >= 3:
                _, p1 = stats.shapiro(group1)
                assumptions['normality_g1'] = p1 > 0.05
            else:
                assumptions['normality_g1'] = None

            if len(group2) >= 3:
                _, p2 = stats.shapiro(group2)
                assumptions['normality_g2'] = p2 > 0.05
            else:
                assumptions['normality_g2'] = None

            # Equal variance test (Levene's test)
            if len(group1) >= 2 and len(group2) >= 2:
                _, p_levene = stats.levene(group1, group2)
                assumptions['equal_variance'] = p_levene > 0.05
            else:
                assumptions['equal_variance'] = None

            return assumptions

        def anova_analysis(self, groups, group_labels=None):
            """Perform one-way ANOVA with post-hoc tests."""
            if group_labels is None:
                group_labels = [f'Group_{i+1}' for i in range(len(groups))]

            # One-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups)

            results = {
                'anova': {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                },
                'groups': group_labels,
                'n_groups': len(groups)
            }

            # Effect size (eta-squared)
            # Calculate sum of squares
            grand_mean = np.mean(np.concatenate(groups))
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2
                           for group in groups)
            ss_total = sum(np.sum((group - grand_mean)**2) for group in groups)

            eta_squared = ss_between / ss_total
            results['anova']['eta_squared'] = eta_squared

            # Post-hoc pairwise comparisons (if significant)
            if p_value < self.alpha:
                results['posthoc'] = self._posthoc_tests(groups, group_labels)

            return results

        def _posthoc_tests(self, groups, group_labels):
            """Perform pairwise post-hoc tests with Bonferroni correction."""
            from itertools import combinations

            posthoc_results = []
            n_comparisons = len(list(combinations(range(len(groups)), 2)))
            bonferroni_alpha = self.alpha / n_comparisons

            for i, j in combinations(range(len(groups)), 2):
                result = self.compare_two_groups(groups[i], groups[j])
                result['comparison'] = f"{group_labels[i]} vs {group_labels[j]}"
                result['bonferroni_alpha'] = bonferroni_alpha
                result['bonferroni_significant'] = result['p_value'] < bonferroni_alpha
                posthoc_results.append(result)

            return posthoc_results

    # Example: Compare sorting efficiency across different morphogen types
    # Generate sample data
    bmp_efficiency = np.random.beta(3, 2, 30) * 100  # Higher efficiency
    wnt_efficiency = np.random.beta(2, 3, 30) * 100  # Lower efficiency
    fgf_efficiency = np.random.beta(2.5, 2.5, 30) * 100  # Medium efficiency

    tester = MorphogenesisTests(alpha=0.05)

    # Two-group comparison
    comparison = tester.compare_two_groups(bmp_efficiency, wnt_efficiency)
    print("Two-group comparison (BMP vs Wnt):")
    print(f"Test: {comparison['test_name']}")
    print(f"p-value: {comparison['p_value']:.6f}")
    print(f"Significant: {comparison['significant']}")
    print(f"Effect size: {comparison['effect_size']:.3f}")

    # ANOVA for multiple groups
    anova_results = tester.anova_analysis(
        [bmp_efficiency, wnt_efficiency, fgf_efficiency],
        ['BMP', 'Wnt', 'FGF']
    )

    print(f"\nANOVA Results:")
    print(f"F-statistic: {anova_results['anova']['f_statistic']:.3f}")
    print(f"p-value: {anova_results['anova']['p_value']:.6f}")
    print(f"Effect size (η²): {anova_results['anova']['eta_squared']:.3f}")

Advanced Statistical Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement sophisticated statistical models for morphogenesis:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import statsmodels.api as sm
    from scipy import stats

    class MorphogenesisModeling:
        """Advanced statistical modeling for morphogenesis research."""

        def __init__(self):
            self.models = {}
            self.scalers = {}

        def multiple_regression(self, data, dependent_var, independent_vars):
            """Perform multiple linear regression analysis."""
            # Prepare data
            y = data[dependent_var].values
            X = data[independent_vars].values

            # Check for multicollinearity
            correlation_matrix = data[independent_vars].corr()
            vif_data = self._calculate_vif(data[independent_vars])

            # Fit model using statsmodels for detailed output
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()

            # Store results
            results = {
                'model': model,
                'summary': model.summary(),
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'coefficients': dict(zip(['intercept'] + independent_vars,
                                       model.params)),
                'p_values': dict(zip(['intercept'] + independent_vars,
                                   model.pvalues)),
                'confidence_intervals': dict(zip(['intercept'] + independent_vars,
                                               model.conf_int().values)),
                'multicollinearity': {
                    'correlation_matrix': correlation_matrix,
                    'vif': vif_data
                },
                'residuals': model.resid,
                'fitted_values': model.fittedvalues
            }

            # Diagnostic tests
            results['diagnostics'] = self._regression_diagnostics(model, X, y)

            return results

        def _calculate_vif(self, data):
            """Calculate Variance Inflation Factor for multicollinearity."""
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            vif_data = pd.DataFrame()
            vif_data["Variable"] = data.columns
            vif_data["VIF"] = [variance_inflation_factor(data.values, i)
                             for i in range(data.shape[1])]
            return vif_data

        def _regression_diagnostics(self, model, X, y):
            """Perform regression diagnostic tests."""
            diagnostics = {}

            # Durbin-Watson test for autocorrelation
            from statsmodels.stats.diagnostic import durbin_watson
            diagnostics['durbin_watson'] = durbin_watson(model.resid)

            # Breusch-Pagan test for heteroscedasticity
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(model.resid,
                                                                    model.model.exog)
            diagnostics['breusch_pagan'] = {
                'statistic': bp_stat,
                'p_value': bp_pvalue
            }

            # Jarque-Bera test for normality of residuals
            from statsmodels.stats.diagnostic import jarque_bera
            jb_stat, jb_pvalue, jb_skew, jb_kurtosis = jarque_bera(model.resid)
            diagnostics['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_pvalue,
                'skewness': jb_skew,
                'kurtosis': jb_kurtosis
            }

            return diagnostics

        def polynomial_regression(self, x, y, degree=2, cv_folds=5):
            """Perform polynomial regression with cross-validation."""
            # Generate polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly_features.fit_transform(x.reshape(-1, 1))

            # Fit model
            model = LinearRegression()
            model.fit(X_poly, y)

            # Cross-validation
            cv_scores = cross_val_score(model, X_poly, y, cv=cv_folds,
                                      scoring='neg_mean_squared_error')

            # Predictions
            y_pred = model.predict(X_poly)

            results = {
                'model': model,
                'polynomial_features': poly_features,
                'degree': degree,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r_squared': r2_score(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'cv_scores': -cv_scores,  # Convert back to positive RMSE
                'cv_mean': -cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }

            return results

        def logistic_regression_morphogenesis(self, data, dependent_var,
                                           independent_vars):
            """Logistic regression for binary morphogenesis outcomes."""
            # Prepare data
            y = data[dependent_var].values
            X = data[independent_vars].values

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            # Fit logistic regression
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_prob_test = model.predict_proba(X_test)[:, 1]

            # Model evaluation
            from sklearn.metrics import (classification_report, confusion_matrix,
                                       roc_auc_score, roc_curve)

            results = {
                'model': model,
                'scaler': scaler,
                'coefficients': dict(zip(independent_vars, model.coef_[0])),
                'intercept': model.intercept_[0],
                'train_accuracy': model.score(X_train, y_train),
                'test_accuracy': model.score(X_test, y_test),
                'classification_report': classification_report(y_test, y_pred_test),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test),
                'roc_auc': roc_auc_score(y_test, y_prob_test),
                'predictions': y_pred_test,
                'probabilities': y_prob_test
            }

            # ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
            results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

            return results

    # Example: Model sorting efficiency based on multiple factors
    np.random.seed(42)
    n_samples = 200

    # Generate synthetic morphogenesis data
    morphogen_conc = np.random.uniform(0.1, 2.0, n_samples)
    cell_density = np.random.uniform(100, 1000, n_samples)
    temperature = np.random.normal(37, 2, n_samples)
    ph_level = np.random.normal(7.4, 0.3, n_samples)

    # Generate dependent variable with some realistic relationships
    sorting_efficiency = (
        30 +
        20 * np.log(morphogen_conc) +
        0.05 * cell_density +
        2 * (temperature - 37) +
        10 * (ph_level - 7.4) +
        np.random.normal(0, 5, n_samples)
    )

    # Ensure efficiency is between 0 and 100
    sorting_efficiency = np.clip(sorting_efficiency, 0, 100)

    research_data = pd.DataFrame({
        'sorting_efficiency': sorting_efficiency,
        'morphogen_concentration': morphogen_conc,
        'cell_density': cell_density,
        'temperature': temperature,
        'ph_level': ph_level
    })

    # Perform multiple regression
    modeler = MorphogenesisModeling()
    regression_results = modeler.multiple_regression(
        research_data,
        'sorting_efficiency',
        ['morphogen_concentration', 'cell_density', 'temperature', 'ph_level']
    )

    print("Multiple Regression Results:")
    print(f"R-squared: {regression_results['r_squared']:.4f}")
    print(f"Adjusted R-squared: {regression_results['adj_r_squared']:.4f}")
    print(f"F-statistic: {regression_results['f_statistic']:.3f}")
    print(f"F p-value: {regression_results['f_pvalue']:.6f}")

    print("\nCoefficients and p-values:")
    for var in regression_results['coefficients']:
        coef = regression_results['coefficients'][var]
        pval = regression_results['p_values'][var]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"{var}: {coef:.4f} (p={pval:.6f}) {sig}")

Data Visualization for Research
-------------------------------

Professional visualization for research publications:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Ellipse
    from scipy import stats

    class ResearchVisualization:
        """Publication-quality visualizations for morphogenesis research."""

        def __init__(self, style='whitegrid', context='paper', palette='Set2'):
            sns.set_style(style)
            sns.set_context(context)
            sns.set_palette(palette)
            plt.rcParams['figure.dpi'] = 300  # High DPI for publications
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.labelsize'] = 12
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['legend.fontsize'] = 10

        def publication_scatter(self, data, x_col, y_col, group_col=None,
                              title="", xlabel="", ylabel="", figsize=(8, 6)):
            """Create publication-quality scatter plot."""
            fig, ax = plt.subplots(figsize=figsize)

            if group_col is not None:
                groups = data[group_col].unique()
                colors = sns.color_palette("Set2", len(groups))

                for i, group in enumerate(groups):
                    group_data = data[data[group_col] == group]
                    ax.scatter(group_data[x_col], group_data[y_col],
                             c=[colors[i]], label=group, alpha=0.7, s=60)

                ax.legend(title=group_col, frameon=True, fancybox=True, shadow=True)
            else:
                ax.scatter(data[x_col], data[y_col], alpha=0.7, s=60)

            # Add regression line if no grouping
            if group_col is None:
                z = np.polyfit(data[x_col], data[y_col], 1)
                p = np.poly1d(z)
                ax.plot(data[x_col], p(data[x_col]), "r--", alpha=0.8, linewidth=2)

                # Calculate and display R²
                correlation = np.corrcoef(data[x_col], data[y_col])[0, 1]
                r_squared = correlation ** 2
                ax.text(0.05, 0.95, f'R² = {r_squared:.3f}',
                       transform=ax.transAxes, bbox=dict(boxstyle="round",
                       facecolor='white', alpha=0.8))

            ax.set_xlabel(xlabel if xlabel else x_col)
            ax.set_ylabel(ylabel if ylabel else y_col)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig, ax

        def error_bar_plot(self, data, x_col, y_col, group_col,
                          error_type='se', title="", xlabel="", ylabel="",
                          figsize=(10, 6)):
            """Create error bar plot with statistical annotations."""
            fig, ax = plt.subplots(figsize=figsize)

            # Calculate statistics for each group
            grouped_stats = data.groupby([x_col, group_col])[y_col].agg([
                'mean', 'std', 'count', 'sem'
            ]).reset_index()

            groups = grouped_stats[group_col].unique()
            x_positions = grouped_stats[x_col].unique()

            # Create bar positions
            bar_width = 0.35
            positions = {}
            for i, group in enumerate(groups):
                positions[group] = np.arange(len(x_positions)) + i * bar_width

            # Plot bars with error bars
            for group in groups:
                group_data = grouped_stats[grouped_stats[group_col] == group]

                error_values = group_data['sem'] if error_type == 'se' else group_data['std']

                ax.bar(positions[group], group_data['mean'], bar_width,
                      yerr=error_values, capsize=5, label=group,
                      alpha=0.8, edgecolor='black', linewidth=0.5)

            # Customize plot
            ax.set_xlabel(xlabel if xlabel else x_col)
            ax.set_ylabel(ylabel if ylabel else f'{y_col} (mean ± {error_type.upper()})')
            ax.set_title(title)
            ax.set_xticks(np.arange(len(x_positions)) + bar_width / 2)
            ax.set_xticklabels(x_positions)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            return fig, ax

        def correlation_heatmap(self, data, variables=None, method='pearson',
                              title="Correlation Matrix", figsize=(10, 8)):
            """Create correlation heatmap with significance annotations."""
            if variables is None:
                variables = data.select_dtypes(include=[np.number]).columns

            correlation_data = data[variables]

            # Calculate correlation matrix
            if method == 'pearson':
                corr_matrix = correlation_data.corr(method='pearson')
            elif method == 'spearman':
                corr_matrix = correlation_data.corr(method='spearman')

            # Calculate p-values
            n = len(correlation_data)
            p_matrix = np.zeros_like(corr_matrix)

            for i, col1 in enumerate(variables):
                for j, col2 in enumerate(variables):
                    if i != j:
                        if method == 'pearson':
                            _, p_val = stats.pearsonr(correlation_data[col1],
                                                    correlation_data[col2])
                        elif method == 'spearman':
                            _, p_val = stats.spearmanr(correlation_data[col1],
                                                     correlation_data[col2])
                        p_matrix[i, j] = p_val

            # Create significance annotation
            significance = np.where(p_matrix < 0.001, '***',
                          np.where(p_matrix < 0.01, '**',
                          np.where(p_matrix < 0.05, '*', '')))

            # Create heatmap
            fig, ax = plt.subplots(figsize=figsize)

            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle

            heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
                                center=0, square=True, cmap='RdBu_r',
                                cbar_kws={"shrink": .8}, ax=ax)

            # Add significance stars
            for i in range(len(variables)):
                for j in range(len(variables)):
                    if not mask[i, j] and significance[i, j]:
                        ax.text(j + 0.7, i + 0.3, significance[i, j],
                               ha='center', va='center', fontsize=12,
                               fontweight='bold', color='black')

            ax.set_title(f'{title} ({method.capitalize()})', pad=20)
            plt.tight_layout()
            return fig, ax

        def time_series_plot(self, data, time_col, value_col, group_col=None,
                           confidence_interval=True, title="", xlabel="Time",
                           ylabel="", figsize=(12, 6)):
            """Create time series plot with confidence intervals."""
            fig, ax = plt.subplots(figsize=figsize)

            if group_col is not None:
                groups = data[group_col].unique()

                for group in groups:
                    group_data = data[data[group_col] == group].copy()
                    group_data = group_data.sort_values(time_col)

                    # Calculate rolling mean and confidence interval
                    if confidence_interval and len(group_data) > 10:
                        # Simple moving average with confidence bands
                        window = max(3, len(group_data) // 10)
                        rolling_mean = group_data[value_col].rolling(window=window,
                                                                   center=True).mean()
                        rolling_std = group_data[value_col].rolling(window=window,
                                                                  center=True).std()

                        ax.fill_between(group_data[time_col],
                                       rolling_mean - 1.96 * rolling_std,
                                       rolling_mean + 1.96 * rolling_std,
                                       alpha=0.2)

                    ax.plot(group_data[time_col], group_data[value_col],
                           label=group, linewidth=2, marker='o', markersize=4)
            else:
                data_sorted = data.sort_values(time_col)
                ax.plot(data_sorted[time_col], data_sorted[value_col],
                       linewidth=2, marker='o', markersize=4)

                if confidence_interval and len(data) > 10:
                    window = max(3, len(data) // 10)
                    rolling_mean = data_sorted[value_col].rolling(window=window,
                                                               center=True).mean()
                    rolling_std = data_sorted[value_col].rolling(window=window,
                                                              center=True).std()

                    ax.fill_between(data_sorted[time_col],
                                   rolling_mean - 1.96 * rolling_std,
                                   rolling_mean + 1.96 * rolling_std,
                                   alpha=0.2)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel if ylabel else value_col)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            if group_col is not None:
                ax.legend()

            plt.tight_layout()
            return fig, ax

    # Example usage for morphogenesis research visualization
    np.random.seed(42)

    # Generate sample data
    n_samples = 150
    research_viz_data = pd.DataFrame({
        'morphogen_concentration': np.random.uniform(0.1, 2.0, n_samples),
        'sorting_efficiency': np.random.beta(2, 3, n_samples) * 100,
        'cell_type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples),
        'treatment': np.random.choice(['Control', 'Treatment'], n_samples),
        'time': np.random.uniform(0, 24, n_samples)
    })

    # Add some correlation
    research_viz_data['sorting_efficiency'] += (
        20 * np.log(research_viz_data['morphogen_concentration']) +
        np.random.normal(0, 5, n_samples)
    )
    research_viz_data['sorting_efficiency'] = np.clip(
        research_viz_data['sorting_efficiency'], 0, 100
    )

    # Create visualizations
    viz = ResearchVisualization()

    # Scatter plot with groups
    fig1, ax1 = viz.publication_scatter(
        research_viz_data, 'morphogen_concentration', 'sorting_efficiency',
        group_col='cell_type',
        title='Cell Sorting Efficiency vs Morphogen Concentration',
        xlabel='Morphogen Concentration (μg/mL)',
        ylabel='Sorting Efficiency (%)'
    )

    # Error bar plot
    fig2, ax2 = viz.error_bar_plot(
        research_viz_data, 'cell_type', 'sorting_efficiency', 'treatment',
        title='Sorting Efficiency by Cell Type and Treatment',
        xlabel='Cell Type',
        ylabel='Sorting Efficiency (%)'
    )

    plt.show()

Research Workflow Integration
-----------------------------

Complete workflow for morphogenesis research projects:

.. code-block:: python

    import os
    import json
    import datetime
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from morphogenesis.experiments import ExperimentRunner
    from morphogenesis.analysis import ComprehensiveAnalyzer

    class ResearchProjectManager:
        """Comprehensive research project management for morphogenesis studies."""

        def __init__(self, project_name, base_dir="./research_projects"):
            self.project_name = project_name
            self.base_dir = Path(base_dir)
            self.project_dir = self.base_dir / project_name

            self.create_project_structure()
            self.load_project_metadata()

        def create_project_structure(self):
            """Create standardized research project directory structure."""
            directories = [
                'data/raw',
                'data/processed',
                'data/results',
                'analysis/descriptive',
                'analysis/inferential',
                'analysis/modeling',
                'figures/exploratory',
                'figures/publication',
                'reports/interim',
                'reports/final',
                'notebooks',
                'src/experiments',
                'src/analysis',
                'references',
                'documentation'
            ]

            for directory in directories:
                (self.project_dir / directory).mkdir(parents=True, exist_ok=True)

            # Create project metadata file
            metadata_file = self.project_dir / 'project_metadata.json'
            if not metadata_file.exists():
                metadata = {
                    'project_name': self.project_name,
                    'created_date': datetime.datetime.now().isoformat(),
                    'description': '',
                    'investigators': [],
                    'research_questions': [],
                    'hypotheses': [],
                    'experiments': [],
                    'analysis_completed': [],
                    'publications': []
                }

                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

        def load_project_metadata(self):
            """Load project metadata from JSON file."""
            metadata_file = self.project_dir / 'project_metadata.json'
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)

        def save_project_metadata(self):
            """Save project metadata to JSON file."""
            metadata_file = self.project_dir / 'project_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)

        def add_experiment(self, experiment_config, description=""):
            """Add new experiment to project."""
            experiment_id = f"exp_{len(self.metadata['experiments']) + 1:03d}"

            experiment_record = {
                'experiment_id': experiment_id,
                'date_created': datetime.datetime.now().isoformat(),
                'description': description,
                'config': experiment_config,
                'status': 'planned',
                'data_files': [],
                'analysis_files': [],
                'results_summary': {}
            }

            self.metadata['experiments'].append(experiment_record)
            self.save_project_metadata()

            # Create experiment directory
            exp_dir = self.project_dir / 'src' / 'experiments' / experiment_id
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Save experiment configuration
            config_file = exp_dir / 'config.json'
            with open(config_file, 'w') as f:
                json.dump(experiment_config, f, indent=2)

            return experiment_id

        def run_experiment(self, experiment_id, runner_class=None):
            """Execute experiment and save results."""
            # Find experiment
            experiment = None
            for exp in self.metadata['experiments']:
                if exp['experiment_id'] == experiment_id:
                    experiment = exp
                    break

            if experiment is None:
                raise ValueError(f"Experiment {experiment_id} not found")

            # Load configuration
            exp_dir = self.project_dir / 'src' / 'experiments' / experiment_id
            config_file = exp_dir / 'config.json'
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Run experiment
            if runner_class is None:
                runner_class = ExperimentRunner

            runner = runner_class(config)
            results = runner.run()

            # Save raw data
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            data_file = self.project_dir / 'data' / 'raw' / f'{experiment_id}_{timestamp}.csv'
            results['data'].to_csv(data_file, index=False)

            # Update experiment record
            experiment['status'] = 'completed'
            experiment['data_files'].append(str(data_file.relative_to(self.project_dir)))
            experiment['results_summary'] = results['summary']
            experiment['date_completed'] = datetime.datetime.now().isoformat()

            self.save_project_metadata()

            return results

        def analyze_experiment(self, experiment_id, analysis_type='comprehensive'):
            """Perform statistical analysis on experiment results."""
            # Find experiment
            experiment = None
            for exp in self.metadata['experiments']:
                if exp['experiment_id'] == experiment_id:
                    experiment = exp
                    break

            if experiment is None:
                raise ValueError(f"Experiment {experiment_id} not found")

            # Load data
            if not experiment['data_files']:
                raise ValueError(f"No data files found for experiment {experiment_id}")

            latest_data_file = self.project_dir / experiment['data_files'][-1]
            data = pd.read_csv(latest_data_file)

            # Perform analysis
            analyzer = ComprehensiveAnalyzer(data)

            if analysis_type == 'comprehensive':
                analysis_results = analyzer.full_analysis()
            elif analysis_type == 'descriptive':
                analysis_results = analyzer.descriptive_analysis()
            elif analysis_type == 'inferential':
                analysis_results = analyzer.inferential_analysis()

            # Save analysis results
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_file = (self.project_dir / 'analysis' / analysis_type /
                           f'{experiment_id}_{analysis_type}_{timestamp}.json')

            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(analysis_results)

            with open(analysis_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            # Update experiment record
            experiment['analysis_files'].append(str(analysis_file.relative_to(self.project_dir)))
            self.save_project_metadata()

            return analysis_results

        def _make_json_serializable(self, obj):
            """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
            if isinstance(obj, dict):
                return {key: self._make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [self._make_json_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            else:
                return obj

        def generate_report(self, experiment_ids=None, report_type='interim'):
            """Generate comprehensive research report."""
            if experiment_ids is None:
                experiment_ids = [exp['experiment_id'] for exp in self.metadata['experiments']]

            report_content = {
                'project_name': self.project_name,
                'report_type': report_type,
                'generated_date': datetime.datetime.now().isoformat(),
                'experiments_included': experiment_ids,
                'summary': {},
                'detailed_results': {}
            }

            # Compile results from all experiments
            for exp_id in experiment_ids:
                experiment = None
                for exp in self.metadata['experiments']:
                    if exp['experiment_id'] == exp_id:
                        experiment = exp
                        break

                if experiment and experiment['status'] == 'completed':
                    report_content['detailed_results'][exp_id] = {
                        'description': experiment['description'],
                        'config': experiment['config'],
                        'results_summary': experiment['results_summary'],
                        'analysis_files': experiment['analysis_files']
                    }

            # Save report
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = (self.project_dir / 'reports' / report_type /
                          f'research_report_{timestamp}.json')

            with open(report_file, 'w') as f:
                json.dump(report_content, f, indent=2)

            return report_content

        def list_experiments(self):
            """List all experiments in the project."""
            if not self.metadata['experiments']:
                print("No experiments found in this project.")
                return

            print(f"\nExperiments in project '{self.project_name}':")
            print("-" * 80)

            for exp in self.metadata['experiments']:
                print(f"ID: {exp['experiment_id']}")
                print(f"Description: {exp['description']}")
                print(f"Status: {exp['status']}")
                print(f"Created: {exp['date_created']}")
                if exp['status'] == 'completed':
                    print(f"Completed: {exp['date_completed']}")
                print("-" * 40)

    # Example usage: Complete research workflow
    # Create new research project
    project = ResearchProjectManager("morphogen_gradient_study")

    # Add research hypothesis and questions to metadata
    project.metadata['research_questions'] = [
        "How does morphogen concentration gradient affect cell sorting patterns?",
        "What is the optimal gradient steepness for maximum sorting efficiency?"
    ]

    project.metadata['hypotheses'] = [
        "Steeper morphogen gradients will result in more efficient cell sorting",
        "There exists an optimal gradient steepness that maximizes sorting while minimizing energy expenditure"
    ]

    project.save_project_metadata()

    # Design experiment
    experiment_config = {
        'experiment_type': 'gradient_analysis',
        'parameters': {
            'gradient_steepness': [0.1, 0.5, 1.0, 2.0, 5.0],
            'morphogen_concentration': [0.5, 1.0, 2.0],
            'simulation_time': 1000,
            'replicates': 10
        },
        'measurements': [
            'sorting_efficiency',
            'convergence_time',
            'energy_dissipation'
        ]
    }

    # Add experiment to project
    exp_id = project.add_experiment(
        experiment_config,
        description="Analysis of morphogen gradient effects on cell sorting efficiency"
    )

    print(f"Created experiment: {exp_id}")
    print("Experiment configuration saved.")
    print(f"Project directory: {project.project_dir}")

Conclusion
----------

This research methodology tutorial provides a comprehensive framework for conducting rigorous morphogenesis research. Key takeaways include:

**Experimental Design**
- Proper hypothesis formation with testable predictions
- Power analysis for appropriate sample sizes
- Randomized controlled designs with proper controls

**Statistical Analysis**
- Comprehensive descriptive statistics
- Appropriate hypothesis testing methods
- Advanced modeling techniques for complex relationships

**Visualization**
- Publication-quality figures with proper error representation
- Clear communication of statistical results
- Professional formatting for research dissemination

**Project Management**
- Standardized directory structure for reproducible research
- Automated data management and analysis workflows
- Comprehensive documentation and reporting systems

**Best Practices**
- Always check statistical assumptions before applying tests
- Use appropriate effect size measures alongside p-values
- Implement proper multiple comparison corrections
- Maintain detailed documentation throughout the research process
- Design experiments with sufficient power to detect meaningful effects

This framework ensures that morphogenesis research meets the highest standards of scientific rigor and reproducibility.