Experimental Design for Morphogenesis Research
=============================================

Rigorous experimental design is the foundation of meaningful morphogenesis research. This guide provides comprehensive methodologies for designing, conducting, and analyzing morphogenesis experiments using the Enhanced Morphogenesis Research Platform.

.. note::
   This methodology follows established practices from experimental biology, adapted for computational morphogenesis research. All approaches are designed to meet standards for peer-reviewed publication.

Principles of Experimental Design
---------------------------------

**Core Principles**

Scientific experiments in morphogenesis research must adhere to these fundamental principles:

**1. Falsifiability**
   * Every hypothesis must be testable and potentially disprovable
   * Design experiments that can distinguish between competing explanations
   * Include conditions that would demonstrate if your hypothesis is wrong

**2. Controlled Variables**
   * Control all variables except those being tested
   * Use appropriate control conditions for comparison
   * Account for confounding factors that might affect results

**3. Reproducibility**
   * Document all parameters and procedures completely
   * Use deterministic simulations with controlled random seeds
   * Design experiments that others can replicate

**4. Statistical Power**
   * Ensure adequate sample sizes to detect meaningful effects
   * Plan statistical analyses before data collection
   * Account for multiple comparisons when testing multiple hypotheses

**5. Biological Relevance**
   * Ground experiments in biological theory and observations
   * Use biologically plausible parameters
   * Interpret results in biological context

Types of Experimental Designs
-----------------------------

**1. Hypothesis Testing Experiments**

Designed to test specific biological hypotheses:

.. code-block:: python

   class HypothesisTestingExperiment:
       """Framework for hypothesis-driven morphogenesis experiments."""

       def __init__(self, hypothesis, predictions, controls):
           self.hypothesis = hypothesis
           self.predictions = predictions
           self.controls = controls

       def design_experiment(self):
           """Design experiment to test hypothesis."""

           experimental_design = {
               'primary_hypothesis': self.hypothesis,
               'null_hypothesis': self.formulate_null_hypothesis(),
               'predictions': self.predictions,
               'independent_variables': self.identify_independent_variables(),
               'dependent_variables': self.identify_dependent_variables(),
               'controlled_variables': self.identify_controlled_variables(),
               'experimental_conditions': self.design_conditions(),
               'control_conditions': self.design_controls(),
               'sample_size': self.calculate_required_sample_size(),
               'statistical_tests': self.plan_statistical_analysis()
           }

           return experimental_design

   # Example: Testing differential adhesion hypothesis
   hypothesis = "Cells with higher adhesion to similar cell types will sort more efficiently"

   predictions = [
       "Increasing adhesion strength difference will increase sorting efficiency",
       "Cells with identical adhesion will not sort",
       "Sorting efficiency will saturate at high adhesion differences"
   ]

   controls = [
       "Identical adhesion for all cell types (negative control)",
       "Random cell movement without adhesion (baseline control)",
       "Known sorting system for validation (positive control)"
   ]

   experiment = HypothesisTestingExperiment(hypothesis, predictions, controls)
   design = experiment.design_experiment()

**2. Exploratory Experiments**

Designed to discover new phenomena or relationships:

.. code-block:: python

   class ExploratoryExperiment:
       """Framework for exploratory morphogenesis research."""

       def __init__(self, research_domain, parameter_space):
           self.domain = research_domain
           self.parameter_space = parameter_space

       def design_systematic_exploration(self):
           """Design systematic parameter space exploration."""

           exploration_design = {
               'parameter_dimensions': list(self.parameter_space.keys()),
               'parameter_ranges': self.parameter_space,
               'sampling_strategy': self.choose_sampling_strategy(),
               'coverage_criteria': self.define_coverage_criteria(),
               'stopping_criteria': self.define_stopping_criteria(),
               'pattern_detection': self.define_pattern_detection_methods(),
               'hypothesis_generation': self.define_hypothesis_generation_procedures()
           }

           return exploration_design

       def choose_sampling_strategy(self):
           """Choose appropriate sampling strategy for parameter space."""

           # Options based on parameter space characteristics
           if self.estimate_computational_cost() > 1000:
               return 'latin_hypercube_sampling'  # Efficient for high-dimensional spaces
           elif self.has_known_nonlinearities():
               return 'adaptive_sampling'         # Focus on interesting regions
           else:
               return 'grid_sampling'             # Systematic coverage

   # Example: Exploring pattern formation parameter space
   parameter_space = {
       'diffusion_rate': (0.01, 1.0),
       'reaction_rate': (0.1, 10.0),
       'inhibitor_strength': (0.0, 5.0),
       'domain_size': (10, 100),
       'initial_noise_level': (0.0, 0.5)
   }

   exploration = ExploratoryExperiment('pattern_formation', parameter_space)
   design = exploration.design_systematic_exploration()

**3. Comparative Studies**

Designed to compare different algorithms, models, or conditions:

.. code-block:: python

   class ComparativeStudy:
       """Framework for comparative morphogenesis studies."""

       def __init__(self, comparison_targets, evaluation_criteria):
           self.targets = comparison_targets
           self.criteria = evaluation_criteria

       def design_comparison(self):
           """Design comparative study."""

           comparison_design = {
               'comparison_targets': self.targets,
               'evaluation_criteria': self.criteria,
               'matched_conditions': self.design_matched_conditions(),
               'control_variables': self.identify_control_variables(),
               'randomization_scheme': self.design_randomization(),
               'blocking_factors': self.identify_blocking_factors(),
               'sample_size_per_group': self.calculate_group_sample_sizes(),
               'statistical_comparisons': self.plan_comparison_statistics()
           }

           return comparison_design

   # Example: Comparing cell sorting algorithms
   algorithms = ['differential_adhesion', 'chemotaxis_based', 'mechanical_forces']

   criteria = [
       'sorting_efficiency',
       'convergence_time',
       'robustness_to_noise',
       'computational_performance'
   ]

   comparison = ComparativeStudy(algorithms, criteria)
   design = comparison.design_comparison()

**4. Dose-Response Studies**

Investigating relationships between parameter values and outcomes:

.. code-block:: python

   class DoseResponseStudy:
       """Framework for dose-response studies in morphogenesis."""

       def __init__(self, parameter, response_variable, expected_relationship):
           self.parameter = parameter
           self.response = response_variable
           self.expected_relationship = expected_relationship

       def design_dose_response(self):
           """Design dose-response study."""

           design = {
               'parameter': self.parameter,
               'response_variable': self.response,
               'dose_levels': self.design_dose_levels(),
               'dose_spacing': self.choose_dose_spacing(),
               'replicates_per_dose': self.calculate_replicates_needed(),
               'control_conditions': self.design_controls(),
               'curve_fitting_approach': self.plan_curve_fitting(),
               'statistical_analysis': self.plan_dose_response_analysis()
           }

           return design

       def design_dose_levels(self):
           """Design appropriate dose levels."""

           if self.expected_relationship == 'linear':
               return self.create_linear_dose_series()
           elif self.expected_relationship == 'sigmoid':
               return self.create_sigmoid_dose_series()
           elif self.expected_relationship == 'threshold':
               return self.create_threshold_dose_series()
           else:
               return self.create_exploratory_dose_series()

   # Example: Adhesion strength vs. sorting efficiency
   study = DoseResponseStudy(
       parameter='adhesion_strength',
       response_variable='sorting_efficiency',
       expected_relationship='sigmoid'
   )
   design = study.design_dose_response()

Sample Size Determination
-------------------------

**Power Analysis for Morphogenesis Studies**

.. code-block:: python

   import scipy.stats as stats
   from scipy.stats import power

   class MorphogenesisPowerAnalysis:
       """Calculate appropriate sample sizes for morphogenesis studies."""

       def __init__(self, alpha=0.05, power=0.8):
           self.alpha = alpha        # Type I error rate
           self.power = power        # Desired statistical power (1 - Type II error rate)

       def calculate_sample_size_two_groups(self, effect_size, std_dev=1.0):
           """Calculate sample size for comparing two groups."""

           # Effect size in Cohen's d units
           cohens_d = effect_size / std_dev

           # Calculate sample size using power analysis
           analysis = stats.ttest_power(cohens_d, power=self.power, alpha=self.alpha)
           sample_size = analysis.solve_power(effect_size=cohens_d, alpha=self.alpha, power=self.power)

           return int(np.ceil(sample_size))

       def calculate_sample_size_anova(self, effect_size, num_groups):
           """Calculate sample size for ANOVA comparing multiple groups."""

           # Convert effect size to Cohen's f
           cohens_f = effect_size / np.sqrt(2)

           # Calculate sample size for ANOVA
           analysis = stats.anova_power_multiway_ols(
               effect_size=cohens_f,
               nobs=None,
               alpha=self.alpha,
               power=self.power,
               k_groups=num_groups
           )

           sample_size_per_group = analysis.solve_power(
               effect_size=cohens_f,
               alpha=self.alpha,
               power=self.power
           )

           return int(np.ceil(sample_size_per_group))

       def calculate_sample_size_correlation(self, expected_correlation):
           """Calculate sample size for correlation studies."""

           # Fisher's z-transformation
           z_r = 0.5 * np.log((1 + expected_correlation) / (1 - expected_correlation))

           # Critical z-value
           z_alpha = stats.norm.ppf(1 - self.alpha/2)
           z_beta = stats.norm.ppf(self.power)

           # Sample size calculation
           n = ((z_alpha + z_beta) / z_r) ** 2 + 3

           return int(np.ceil(n))

   # Example usage
   power_analysis = MorphogenesisPowerAnalysis(alpha=0.05, power=0.8)

   # For comparing sorting efficiency between two cell types
   n_two_groups = power_analysis.calculate_sample_size_two_groups(
       effect_size=0.5,  # Expected difference in sorting scores
       std_dev=0.2       # Expected standard deviation
   )
   print(f"Required sample size per group: {n_two_groups}")

   # For comparing multiple sorting algorithms
   n_anova = power_analysis.calculate_sample_size_anova(
       effect_size=0.3,  # Expected effect size (Cohen's f)
       num_groups=4      # Number of algorithms to compare
   )
   print(f"Required sample size per algorithm: {n_anova}")

**Effect Size Estimation**

.. code-block:: python

   class EffectSizeEstimation:
       """Estimate appropriate effect sizes for morphogenesis studies."""

       def estimate_from_literature(self, literature_values):
           """Estimate effect size from literature values."""

           # Calculate Cohen's d from means and standard deviations
           mean_diff = abs(literature_values['group1_mean'] - literature_values['group2_mean'])
           pooled_std = np.sqrt(
               (literature_values['group1_std']**2 + literature_values['group2_std']**2) / 2
           )

           cohens_d = mean_diff / pooled_std
           return cohens_d

       def estimate_from_pilot_study(self, pilot_data):
           """Estimate effect size from pilot study data."""

           if len(pilot_data) >= 2:
               effect_size = (np.max(pilot_data) - np.min(pilot_data)) / np.std(pilot_data)
               return effect_size
           else:
               raise ValueError("Need at least 2 data points for pilot study estimation")

       def estimate_minimum_meaningful_effect(self, measurement_precision, biological_relevance):
           """Estimate minimum biologically meaningful effect size."""

           # Minimum detectable effect should be larger than measurement noise
           # and biologically meaningful
           minimum_effect = max(
               2 * measurement_precision,  # 2x measurement error
               biological_relevance        # Biologically meaningful difference
           )

           return minimum_effect

Randomization and Controls
-------------------------

**Randomization Strategies**

.. code-block:: python

   class RandomizationDesign:
       """Design appropriate randomization for morphogenesis experiments."""

       def __init__(self, random_seed=None):
           self.random_seed = random_seed
           if random_seed:
               np.random.seed(random_seed)

       def simple_randomization(self, n_subjects, n_groups):
           """Simple random assignment to groups."""

           group_assignments = np.random.choice(n_groups, size=n_subjects)
           return group_assignments

       def block_randomization(self, n_subjects, n_groups, block_size):
           """Block randomization for balanced group sizes."""

           n_blocks = n_subjects // block_size
           assignments = []

           for _ in range(n_blocks):
               # Create balanced block
               block = np.tile(np.arange(n_groups), block_size // n_groups)
               np.random.shuffle(block)
               assignments.extend(block)

           # Handle remaining subjects
           remaining = n_subjects % block_size
           if remaining > 0:
               final_block = np.random.choice(n_groups, size=remaining)
               assignments.extend(final_block)

           return np.array(assignments)

       def stratified_randomization(self, subjects_data, stratification_variable, n_groups):
           """Stratified randomization based on important variables."""

           assignments = np.zeros(len(subjects_data))

           # Get unique strata
           strata = np.unique(subjects_data[stratification_variable])

           for stratum in strata:
               stratum_indices = np.where(subjects_data[stratification_variable] == stratum)[0]
               stratum_assignments = self.simple_randomization(len(stratum_indices), n_groups)
               assignments[stratum_indices] = stratum_assignments

           return assignments

**Control Design**

.. code-block:: python

   class ControlDesign:
       """Design appropriate controls for morphogenesis experiments."""

       def design_negative_controls(self, experimental_condition):
           """Design negative controls that should show no effect."""

           negative_controls = []

           if 'adhesion_strength' in experimental_condition:
               negative_controls.append({
                   'name': 'no_adhesion_control',
                   'description': 'All adhesion strengths set to zero',
                   'parameters': {'adhesion_strength': 0.0},
                   'expected_result': 'Random distribution, no sorting'
               })

           if 'communication_range' in experimental_condition:
               negative_controls.append({
                   'name': 'no_communication_control',
                   'description': 'Communication range set to zero',
                   'parameters': {'communication_range': 0.0},
                   'expected_result': 'Independent behavior, no coordination'
               })

           return negative_controls

       def design_positive_controls(self, experimental_condition):
           """Design positive controls that should show known effects."""

           positive_controls = []

           if 'cell_sorting' in experimental_condition:
               positive_controls.append({
                   'name': 'strong_sorting_control',
                   'description': 'Very high adhesion differences',
                   'parameters': {'adhesion_difference': 0.9},
                   'expected_result': 'Complete sorting within 100 steps'
               })

           return positive_controls

       def design_vehicle_controls(self, experimental_condition):
           """Design vehicle/baseline controls for comparison."""

           vehicle_controls = []

           # Default parameter settings
           vehicle_controls.append({
               'name': 'default_parameters',
               'description': 'Standard parameter set from literature',
               'parameters': self.get_standard_parameters(),
               'expected_result': 'Baseline behavior from literature'
           })

           return vehicle_controls

Experimental Execution
----------------------

**Experiment Implementation Framework**

.. code-block:: python

   class MorphogenesisExperimentRunner:
       """Framework for executing morphogenesis experiments."""

       def __init__(self, experimental_design, random_seed=42):
           self.design = experimental_design
           self.random_seed = random_seed
           self.results = []

       async def execute_experiment(self):
           """Execute the complete experimental design."""

           print("Starting morphogenesis experiment...")
           print(f"Design: {self.design['experiment_type']}")
           print(f"Conditions: {len(self.design['conditions'])}")

           for condition_idx, condition in enumerate(self.design['conditions']):
               print(f"\\nExecuting condition {condition_idx + 1}/{len(self.design['conditions'])}")
               print(f"Parameters: {condition['parameters']}")

               condition_results = await self.execute_condition(condition)
               condition_results['condition_id'] = condition_idx
               condition_results['condition_parameters'] = condition['parameters']

               self.results.append(condition_results)

           print("\\nExperiment execution complete!")
           return self.results

       async def execute_condition(self, condition):
           """Execute a single experimental condition."""

           condition_data = []

           for replicate in range(condition['replicates']):
               # Create unique seed for this replicate
               replicate_seed = self.random_seed + replicate * 1000 + len(condition_data)

               print(f"  Replicate {replicate + 1}/{condition['replicates']} (seed: {replicate_seed})")

               # Run simulation
               simulation_config = {
                   **condition['parameters'],
                   'random_seed': replicate_seed,
                   'save_snapshots': True,
                   'collect_metrics': True
               }

               experiment = ExperimentRunner(simulation_config)
               replicate_results = await experiment.run()

               # Add metadata
               replicate_results.replicate_id = replicate
               replicate_results.condition_id = condition.get('condition_id', 'unknown')

               condition_data.append(replicate_results)

           return {
               'condition_name': condition.get('name', 'unnamed'),
               'replicates': condition_data,
               'n_replicates': len(condition_data)
           }

**Data Quality Control**

.. code-block:: python

   class DataQualityControl:
       """Quality control checks for morphogenesis experiment data."""

       def __init__(self, quality_thresholds=None):
           self.thresholds = quality_thresholds or self.default_thresholds()

       def default_thresholds(self):
           """Default quality control thresholds."""
           return {
               'min_simulation_steps': 50,
               'max_simulation_time': 3600,  # 1 hour
               'min_agent_survival_rate': 0.8,
               'max_memory_usage_gb': 16,
               'min_convergence_score': 0.9
           }

       def check_data_quality(self, experiment_results):
           """Perform quality control checks on experiment results."""

           quality_report = {
               'passed_qc': True,
               'failed_checks': [],
               'warnings': [],
               'statistics': {}
           }

           for condition_idx, condition in enumerate(experiment_results):
               condition_qc = self.check_condition_quality(condition)

               if not condition_qc['passed_qc']:
                   quality_report['passed_qc'] = False
                   quality_report['failed_checks'].extend([
                       f"Condition {condition_idx}: {check}"
                       for check in condition_qc['failed_checks']
                   ])

               quality_report['warnings'].extend([
                   f"Condition {condition_idx}: {warning}"
                   for warning in condition_qc['warnings']
               ])

           return quality_report

       def check_condition_quality(self, condition_data):
           """Check quality of data from a single condition."""

           qc_results = {
               'passed_qc': True,
               'failed_checks': [],
               'warnings': []
           }

           for replicate_idx, replicate in enumerate(condition_data['replicates']):
               replicate_qc = self.check_replicate_quality(replicate)

               if not replicate_qc['passed_qc']:
                   qc_results['passed_qc'] = False
                   qc_results['failed_checks'].extend([
                       f"Replicate {replicate_idx}: {check}"
                       for check in replicate_qc['failed_checks']
                   ])

           return qc_results

       def check_replicate_quality(self, replicate_data):
           """Check quality of a single replicate."""

           qc = {'passed_qc': True, 'failed_checks': [], 'warnings': []}

           # Check simulation completion
           if replicate_data.simulation_steps < self.thresholds['min_simulation_steps']:
               qc['passed_qc'] = False
               qc['failed_checks'].append("Simulation terminated early")

           # Check execution time
           if replicate_data.execution_time > self.thresholds['max_simulation_time']:
               qc['warnings'].append("Simulation took unusually long")

           # Check agent survival
           if hasattr(replicate_data, 'final_agent_count'):
               survival_rate = replicate_data.final_agent_count / replicate_data.initial_agent_count
               if survival_rate < self.thresholds['min_agent_survival_rate']:
                   qc['warnings'].append(f"Low agent survival rate: {survival_rate:.2f}")

           # Check convergence
           if hasattr(replicate_data, 'convergence_score'):
               if replicate_data.convergence_score < self.thresholds['min_convergence_score']:
                   qc['warnings'].append("Simulation may not have converged")

           return qc

Parameter Sensitivity Analysis
------------------------------

**Systematic Parameter Exploration**

.. code-block:: python

   class ParameterSensitivityAnalysis:
       """Analyze parameter sensitivity in morphogenesis models."""

       def __init__(self, base_parameters, analysis_type='one_at_a_time'):
           self.base_parameters = base_parameters
           self.analysis_type = analysis_type

       def design_sensitivity_study(self, parameters_to_test, variation_range=0.2):
           """Design parameter sensitivity study."""

           sensitivity_conditions = []

           if self.analysis_type == 'one_at_a_time':
               # Vary one parameter at a time
               for param in parameters_to_test:
                   base_value = self.base_parameters[param]

                   # Create variations around base value
                   variations = self.create_parameter_variations(base_value, variation_range)

                   for variation in variations:
                       condition = self.base_parameters.copy()
                       condition[param] = variation

                       sensitivity_conditions.append({
                           'name': f'{param}_variation_{variation}',
                           'parameters': condition,
                           'varied_parameter': param,
                           'varied_value': variation,
                           'base_value': base_value
                       })

           elif self.analysis_type == 'factorial':
               # Full factorial design
               sensitivity_conditions = self.create_factorial_design(
                   parameters_to_test, variation_range
               )

           elif self.analysis_type == 'latin_hypercube':
               # Latin hypercube sampling
               sensitivity_conditions = self.create_lhs_design(
                   parameters_to_test, variation_range
               )

           return sensitivity_conditions

       def create_parameter_variations(self, base_value, variation_range):
           """Create parameter variations around base value."""

           # Create variations: base ± variation_range * base
           variation_amount = base_value * variation_range

           variations = [
               max(0, base_value - variation_amount),  # Lower bound
               base_value,                             # Base value
               base_value + variation_amount           # Upper bound
           ]

           return variations

       def analyze_sensitivity_results(self, sensitivity_results):
           """Analyze results of parameter sensitivity study."""

           sensitivity_analysis = {}

           for param in self.get_varied_parameters(sensitivity_results):
               param_results = self.extract_parameter_results(sensitivity_results, param)

               sensitivity_metrics = {
                   'parameter': param,
                   'sensitivity_coefficient': self.calculate_sensitivity_coefficient(param_results),
                   'effect_size': self.calculate_effect_size(param_results),
                   'statistical_significance': self.test_statistical_significance(param_results),
                   'biological_relevance': self.assess_biological_relevance(param_results)
               }

               sensitivity_analysis[param] = sensitivity_metrics

           return sensitivity_analysis

**Global Sensitivity Analysis**

.. code-block:: python

   class GlobalSensitivityAnalysis:
       """Perform global sensitivity analysis using Sobol indices."""

       def __init__(self, parameter_distributions):
           self.parameter_distributions = parameter_distributions

       def design_sobol_study(self, n_samples=1000):
           """Design Sobol sensitivity analysis study."""

           from SALib.sample import sobol

           # Define parameter problem
           problem = {
               'num_vars': len(self.parameter_distributions),
               'names': list(self.parameter_distributions.keys()),
               'bounds': [dist['bounds'] for dist in self.parameter_distributions.values()]
           }

           # Generate Sobol samples
           param_values = sobol.sample(problem, n_samples)

           # Convert to experimental conditions
           conditions = []
           for i, sample in enumerate(param_values):
               condition_params = {}
               for j, param_name in enumerate(problem['names']):
                   condition_params[param_name] = sample[j]

               conditions.append({
                   'name': f'sobol_sample_{i}',
                   'parameters': condition_params,
                   'sample_id': i
               })

           return conditions, problem

       def analyze_sobol_results(self, experiment_results, problem):
           """Analyze Sobol sensitivity analysis results."""

           from SALib.analyze import sobol

           # Extract output values
           Y = np.array([result['primary_output'] for result in experiment_results])

           # Perform Sobol analysis
           Si = sobol.analyze(problem, Y)

           sensitivity_results = {
               'first_order_indices': dict(zip(problem['names'], Si['S1'])),
               'total_order_indices': dict(zip(problem['names'], Si['ST'])),
               'second_order_indices': Si['S2'],
               'confidence_intervals': {
                   'first_order': dict(zip(problem['names'], Si['S1_conf'])),
                   'total_order': dict(zip(problem['names'], Si['ST_conf']))
               }
           }

           return sensitivity_results

Common Experimental Patterns
----------------------------

**Pattern 1: Algorithm Comparison**

.. code-block:: python

   async def design_algorithm_comparison_study():
       """Design study comparing different morphogenetic algorithms."""

       # Algorithms to compare
       algorithms = {
           'differential_adhesion': {
               'behavior': 'sorting',
               'adhesion_based': True,
               'parameters': {'adhesion_strength': 0.7}
           },
           'chemotaxis_guided': {
               'behavior': 'chemotaxis',
               'chemical_gradient': True,
               'parameters': {'chemotaxis_strength': 0.5}
           },
           'mechanical_forces': {
               'behavior': 'mechanical',
               'force_based': True,
               'parameters': {'force_strength': 1.0}
           }
       }

       # Common experimental conditions
       base_config = {
           'population_size': 200,
           'grid_size': (30, 30),
           'simulation_steps': 300,
           'replicates': 20
       }

       # Create experimental conditions
       conditions = []
       for alg_name, alg_config in algorithms.items():
           condition = {
               'name': f'{alg_name}_algorithm',
               'parameters': {**base_config, **alg_config['parameters']},
               'algorithm_type': alg_name,
               'replicates': base_config['replicates']
           }
           conditions.append(condition)

       experimental_design = {
           'experiment_type': 'algorithm_comparison',
           'conditions': conditions,
           'evaluation_metrics': ['sorting_efficiency', 'convergence_time', 'stability'],
           'statistical_analysis': 'one_way_anova_with_post_hoc'
       }

       return experimental_design

**Pattern 2: Dose-Response Study**

.. code-block:: python

   async def design_dose_response_study():
       """Design dose-response study for parameter effects."""

       # Parameter to vary
       parameter = 'adhesion_strength'
       dose_levels = np.logspace(-2, 0, 8)  # 0.01 to 1.0 in log scale

       # Base experimental configuration
       base_config = {
           'population_size': 150,
           'grid_size': (25, 25),
           'simulation_steps': 250,
           'cell_types': ['A', 'B']
       }

       # Create conditions for each dose level
       conditions = []
       for dose in dose_levels:
           condition = {
               'name': f'{parameter}_{dose:.3f}',
               'parameters': {**base_config, parameter: dose},
               'dose_level': dose,
               'replicates': 15
           }
           conditions.append(condition)

       experimental_design = {
           'experiment_type': 'dose_response',
           'parameter': parameter,
           'dose_levels': dose_levels,
           'conditions': conditions,
           'curve_fitting': 'sigmoid_curve',
           'statistical_analysis': 'nonlinear_regression'
       }

       return experimental_design

**Pattern 3: Multi-Factor Study**

.. code-block:: python

   async def design_multi_factor_study():
       """Design multi-factor study examining parameter interactions."""

       # Factors and their levels
       factors = {
           'adhesion_strength': [0.3, 0.7],
           'population_size': [100, 200],
           'communication_range': [2.0, 4.0]
       }

       # Create full factorial design
       from itertools import product

       factor_combinations = list(product(*factors.values()))
       factor_names = list(factors.keys())

       conditions = []
       for combo_idx, combination in enumerate(factor_combinations):
           parameters = dict(zip(factor_names, combination))

           # Add base parameters
           base_params = {
               'grid_size': (25, 25),
               'simulation_steps': 200,
               'cell_types': ['A', 'B']
           }

           condition = {
               'name': f'combination_{combo_idx}',
               'parameters': {**base_params, **parameters},
               'factor_levels': parameters,
               'replicates': 12
           }
           conditions.append(condition)

       experimental_design = {
           'experiment_type': 'multi_factor',
           'factors': factors,
           'conditions': conditions,
           'statistical_analysis': 'multi_way_anova',
           'interaction_analysis': True
       }

       return experimental_design

Troubleshooting Experimental Issues
-----------------------------------

**Common Problems and Solutions**

.. code-block:: python

   class ExperimentTroubleshooting:
       """Guide for troubleshooting experimental issues."""

       def diagnose_convergence_issues(self, experiment_results):
           """Diagnose and suggest solutions for convergence problems."""

           issues = []
           solutions = []

           # Check for insufficient simulation time
           final_steps = [r.simulation_steps for r in experiment_results]
           if np.mean(final_steps) < 100:
               issues.append("Simulation time too short")
               solutions.append("Increase simulation_steps to at least 200")

           # Check for parameter issues
           organization_scores = [r.final_organization_score for r in experiment_results]
           if np.var(organization_scores) > 0.5:
               issues.append("High variance in final organization")
               solutions.append("Check parameter stability, increase replicates")

           # Check for system size effects
           population_sizes = [r.population_size for r in experiment_results]
           if np.min(population_sizes) < 50:
               issues.append("System too small for stable patterns")
               solutions.append("Increase population size to at least 100")

           return {'issues': issues, 'solutions': solutions}

       def diagnose_performance_issues(self, experiment_results):
           """Diagnose performance problems."""

           performance_issues = []

           # Check execution times
           execution_times = [r.execution_time for r in experiment_results]
           mean_time = np.mean(execution_times)

           if mean_time > 300:  # 5 minutes
               performance_issues.append({
                   'issue': 'Long execution time',
                   'current_time': f'{mean_time:.1f} seconds',
                   'solutions': [
                       'Reduce population size',
                       'Decrease simulation steps',
                       'Optimize neighbor search radius',
                       'Use parallel processing'
                   ]
               })

           # Check memory usage
           if hasattr(experiment_results[0], 'peak_memory_mb'):
               memory_usage = [r.peak_memory_mb for r in experiment_results]
               if np.max(memory_usage) > 8000:  # 8GB
                   performance_issues.append({
                       'issue': 'High memory usage',
                       'peak_memory': f'{np.max(memory_usage):.0f} MB',
                       'solutions': [
                           'Reduce snapshot frequency',
                           'Use data compression',
                           'Implement streaming data export',
                           'Reduce grid resolution'
                       ]
                   })

           return performance_issues

Best Practices Summary
---------------------

**Pre-Experiment Checklist**

.. code-block:: python

   def pre_experiment_checklist():
       """Checklist to complete before running experiments."""

       checklist = {
           'hypothesis_formulation': [
               "Primary hypothesis clearly stated",
               "Null hypothesis defined",
               "Predictions specified",
               "Alternative explanations considered"
           ],
           'experimental_design': [
               "Appropriate controls included",
               "Variables properly controlled",
               "Randomization scheme planned",
               "Sample size calculated with power analysis"
           ],
           'technical_preparation': [
               "Parameters validated with pilot studies",
               "Code tested and debugged",
               "Data collection procedures defined",
               "Quality control measures implemented"
           ],
           'analysis_planning': [
               "Statistical analysis plan defined",
               "Multiple comparison corrections planned",
               "Effect size measures specified",
               "Visualization approach planned"
           ]
       }

       return checklist

**Post-Experiment Checklist**

.. code-block:: python

   def post_experiment_checklist():
       """Checklist for after experiment completion."""

       checklist = {
           'data_quality': [
               "Quality control checks passed",
               "Missing data patterns examined",
               "Outliers identified and handled",
               "Assumptions of statistical tests verified"
           ],
           'analysis_integrity': [
               "All planned analyses completed",
               "Multiple testing corrections applied",
               "Effect sizes calculated and reported",
               "Confidence intervals provided"
           ],
           'interpretation': [
               "Results interpreted in biological context",
               "Limitations clearly acknowledged",
               "Alternative explanations considered",
               "Future research directions identified"
           ],
           'reproducibility': [
               "All parameters and procedures documented",
               "Code and data archived",
               "Analysis scripts provided",
               "Random seeds recorded"
           ]
       }

       return checklist

By following these experimental design principles and methodologies, researchers can conduct rigorous, reproducible morphogenesis studies that contribute meaningful insights to the field. Remember that good experimental design is iterative - use pilot studies to refine your approach, and don't hesitate to modify your design based on initial results.