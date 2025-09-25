Intermediate Tutorials
======================

These tutorials build upon basic morphogenesis concepts to explore more advanced features like multi-scale modeling, optimization techniques, and data analysis workflows.

Tutorial 1: Building Multi-Scale Models
----------------------------------------

Learn how to create models that operate across multiple biological scales, from molecular to tissue level.

**Learning Objectives:**
- Understand multi-scale modeling principles
- Implement gene regulatory networks
- Connect molecular processes to cellular behavior
- Analyze cross-scale interactions

**Prerequisites:**
- Completed beginner tutorials
- Basic understanding of gene regulation
- Familiarity with async programming

Step 1: Setting Up the Multi-Scale Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, let's create a framework that can handle multiple scales of organization.

.. code-block:: python

   import asyncio
   import numpy as np
   from core.coordination.coordinator import DeterministicCoordinator
   from core.agents.cell_agent import AsyncCellAgent

   class MultiScaleSystem:
       def __init__(self, domain_size=(50, 50)):
           self.domain_size = domain_size

           # Different scale coordinators
           self.molecular_coordinator = MolecularCoordinator()
           self.cellular_coordinator = DeterministicCoordinator(
               grid_size=domain_size,
               max_agents=1000
           )
           self.tissue_coordinator = TissueCoordinator()

           # Scale interaction managers
           self.scale_bridges = {
               'molecular_to_cellular': MolecularToCellularBridge(),
               'cellular_to_tissue': CellularToTissueBridge(),
               'tissue_to_cellular': TissueToCellularBridge()
           }

       async def initialize_system(self):
           """Initialize all scales of the system."""
           # Initialize molecular level
           await self.molecular_coordinator.initialize_gene_networks()

           # Initialize cellular level
           cells = await self.create_initial_cell_population()
           await self.cellular_coordinator.add_agents(cells)

           # Initialize tissue level
           await self.tissue_coordinator.initialize_tissue_structure()

       async def step(self, dt=0.1):
           """Coordinated step across all scales."""
           # Step 1: Update molecular level (fastest timescale)
           await self.molecular_coordinator.step(dt * 0.1)

           # Step 2: Transfer molecular -> cellular information
           molecular_state = await self.molecular_coordinator.get_state()
           await self.scale_bridges['molecular_to_cellular'].transfer(
               molecular_state, self.cellular_coordinator
           )

           # Step 3: Update cellular level
           await self.cellular_coordinator.step()

           # Step 4: Transfer cellular -> tissue information
           cellular_state = await self.cellular_coordinator.get_state()
           await self.scale_bridges['cellular_to_tissue'].transfer(
               cellular_state, self.tissue_coordinator
           )

           # Step 5: Update tissue level (slowest timescale)
           await self.tissue_coordinator.step(dt * 10)

           # Step 6: Apply tissue-level constraints to cells
           tissue_constraints = await self.tissue_coordinator.get_constraints()
           await self.scale_bridges['tissue_to_cellular'].apply_constraints(
               tissue_constraints, self.cellular_coordinator
           )

**Exercise:** Create your own `MolecularCoordinator` class that manages gene expression dynamics.

Step 2: Implementing Gene Regulatory Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now let's create a gene regulatory network that influences cellular behavior.

.. code-block:: python

   class GeneRegulatoryNetwork:
       def __init__(self, genes, interactions):
           self.genes = genes  # List of gene names
           self.interactions = interactions  # Dictionary of regulatory relationships
           self.expression_levels = {gene: 0.1 for gene in genes}  # Initial expression

       async def update_expression(self, external_signals=None, dt=0.01):
           """Update gene expression based on regulatory interactions."""
           new_expression = {}

           for gene in self.genes:
               # Base expression level
               expression_change = 0.0

               # Regulatory inputs
               for regulator, effect in self.interactions.get(gene, {}).items():
                   regulator_level = self.expression_levels[regulator]
                   expression_change += effect * regulator_level

               # External signals (morphogens, mechanical forces, etc.)
               if external_signals and gene in external_signals:
                   expression_change += external_signals[gene]

               # Update expression with decay
               current_expression = self.expression_levels[gene]
               new_expression[gene] = current_expression + dt * (
                   expression_change - 0.1 * current_expression  # Decay term
               )

               # Ensure non-negative expression
               new_expression[gene] = max(0.0, new_expression[gene])

           self.expression_levels = new_expression

   class MultiscaleCell(AsyncCellAgent):
       def __init__(self, agent_id, initial_position, cell_type):
           super().__init__(agent_id, initial_position)
           self.cell_type = cell_type

           # Gene regulatory network
           self.grn = self.create_gene_network()

           # Cellular properties influenced by gene expression
           self.adhesion_strength = 0.5
           self.migration_speed = 1.0
           self.proliferation_rate = 0.1

       def create_gene_network(self):
           """Create cell-type specific gene network."""
           genes = ['gene_A', 'gene_B', 'gene_C', 'adhesion_gene', 'migration_gene']

           # Example regulatory interactions
           interactions = {
               'gene_A': {'gene_A': 0.5, 'gene_B': -0.3},  # Self-activation, inhibits B
               'gene_B': {'gene_A': -0.2, 'gene_B': 0.3, 'gene_C': 0.4},
               'gene_C': {'gene_B': 0.5},
               'adhesion_gene': {'gene_A': 0.6, 'gene_B': -0.2},
               'migration_gene': {'gene_C': 0.8, 'adhesion_gene': -0.4}
           }

           return GeneRegulatoryNetwork(genes, interactions)

       async def molecular_to_cellular_coupling(self):
           """Update cellular properties based on gene expression."""
           # Adhesion strength based on adhesion gene
           adhesion_expression = self.grn.expression_levels['adhesion_gene']
           self.adhesion_strength = 0.2 + 0.8 * adhesion_expression

           # Migration speed based on migration gene
           migration_expression = self.grn.expression_levels['migration_gene']
           self.migration_speed = 0.1 + 1.9 * migration_expression

           # Proliferation rate based on gene A
           gene_a_expression = self.grn.expression_levels['gene_A']
           self.proliferation_rate = 0.05 + 0.15 * gene_a_expression

       async def sense_molecular_environment(self):
           """Sense molecular signals that affect gene expression."""
           # This would interface with morphogen fields, mechanical signals, etc.
           external_signals = {}

           # Example: morphogen affecting gene_A
           morphogen_level = await self.sense_morphogen('developmental_signal')
           if morphogen_level:
               external_signals['gene_A'] = morphogen_level * 0.2

           # Example: mechanical stress affecting gene_C
           mechanical_stress = await self.sense_mechanical_stress()
           if mechanical_stress:
               external_signals['gene_C'] = mechanical_stress * 0.1

           return external_signals

       async def step(self, dt=0.1):
           """Multi-scale cell step."""
           # Update gene regulatory network
           external_signals = await self.sense_molecular_environment()
           await self.grn.update_expression(external_signals, dt)

           # Update cellular properties based on gene expression
           await self.molecular_to_cellular_coupling()

           # Execute cellular behaviors
           await self.cellular_behavior_step()

       async def cellular_behavior_step(self):
           """Execute cellular behaviors based on current properties."""
           # Migration decision based on current migration speed
           if self.migration_speed > 0.5:
               migration_direction = await self.calculate_migration_direction()
               await self.move(migration_direction * self.migration_speed)

           # Proliferation decision
           if np.random.random() < self.proliferation_rate * 0.01:
               await self.attempt_division()

**Exercise:** Extend this gene network to include at least 2 more genes and their regulatory interactions.

Step 3: Creating Scale Bridges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scale bridges handle information transfer between different organizational levels.

.. code-block:: python

   class MolecularToCellularBridge:
       """Transfers molecular information to cellular level."""

       async def transfer(self, molecular_state, cellular_coordinator):
           """Transfer molecular state to cellular agents."""
           cells = await cellular_coordinator.get_agents()

           for cell in cells:
               if hasattr(cell, 'grn'):
                   # Update gene expression based on global molecular state
                   await self.update_cell_gene_expression(cell, molecular_state)

       async def update_cell_gene_expression(self, cell, molecular_state):
           """Update individual cell gene expression."""
           # Example: global signaling molecules affect local gene expression
           global_signals = molecular_state.get('global_signals', {})

           for signal_name, signal_level in global_signals.items():
               # Apply signal to specific genes
               if signal_name == 'differentiation_signal':
                   cell.grn.expression_levels['gene_A'] += signal_level * 0.05
               elif signal_name == 'growth_signal':
                   cell.grn.expression_levels['gene_B'] += signal_level * 0.03

   class CellularToTissueBridge:
       """Transfers cellular information to tissue level."""

       async def transfer(self, cellular_state, tissue_coordinator):
           """Transfer cellular state to tissue level."""
           # Aggregate cellular properties
           cell_density_map = await self.create_density_map(cellular_state)
           mechanical_stress_map = await self.calculate_stress_field(cellular_state)

           # Update tissue-level properties
           await tissue_coordinator.update_from_cellular_data({
               'density_map': cell_density_map,
               'stress_field': mechanical_stress_map
           })

       async def create_density_map(self, cellular_state):
           """Create spatial density map from cell positions."""
           # Implementation would create a continuous field from discrete cell positions
           pass

       async def calculate_stress_field(self, cellular_state):
           """Calculate mechanical stress field from cellular forces."""
           # Implementation would compute stress tensor field
           pass

**Exercise:** Implement the missing methods in the bridge classes.

Step 4: Running Multi-Scale Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now let's put everything together in a complete simulation.

.. code-block:: python

   async def run_multiscale_simulation():
       """Complete multi-scale morphogenesis simulation."""
       print("Starting multi-scale simulation...")

       # Initialize system
       system = MultiScaleSystem(domain_size=(60, 60))
       await system.initialize_system()

       # Simulation parameters
       total_time = 100.0  # simulation hours
       dt = 0.1
       timesteps = int(total_time / dt)

       # Data collection
       gene_expression_history = []
       cell_behavior_history = []
       tissue_organization_history = []

       # Main simulation loop
       for timestep in range(timesteps):
           current_time = timestep * dt

           # Step the multi-scale system
           await system.step(dt)

           # Collect data every 10 timesteps
           if timestep % 10 == 0:
               # Collect gene expression data
               cells = await system.cellular_coordinator.get_agents()
               gene_data = await collect_gene_expression_data(cells)
               gene_expression_history.append(gene_data)

               # Collect cellular behavior data
               behavior_data = await collect_behavior_data(cells)
               cell_behavior_history.append(behavior_data)

               # Collect tissue organization data
               tissue_data = await system.tissue_coordinator.get_organization_metrics()
               tissue_organization_history.append(tissue_data)

               print(f"Time: {current_time:.1f}h - "
                     f"Avg migration speed: {behavior_data['avg_migration_speed']:.2f}")

       return {
           'gene_expression': gene_expression_history,
           'cell_behavior': cell_behavior_history,
           'tissue_organization': tissue_organization_history
       }

   async def collect_gene_expression_data(cells):
       """Collect gene expression data from all cells."""
       all_expression = {}

       for cell in cells:
           if hasattr(cell, 'grn'):
               for gene, expression in cell.grn.expression_levels.items():
                   if gene not in all_expression:
                       all_expression[gene] = []
                   all_expression[gene].append(expression)

       # Calculate averages
       avg_expression = {gene: np.mean(values)
                        for gene, values in all_expression.items()}

       return avg_expression

   # Run the simulation
   if __name__ == "__main__":
       results = asyncio.run(run_multiscale_simulation())

**Exercise:** Add visualization code to plot gene expression dynamics over time.

Tutorial 2: Optimization and Parameter Tuning
----------------------------------------------

Learn how to optimize morphogenesis simulations for specific outcomes and tune parameters systematically.

**Learning Objectives:**
- Understand parameter space exploration
- Implement genetic algorithms for optimization
- Use Bayesian optimization for efficient parameter tuning
- Analyze parameter sensitivity

Step 1: Parameter Space Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, define the parameter space for optimization.

.. code-block:: python

   class ParameterSpace:
       def __init__(self):
           self.parameters = {}
           self.constraints = {}

       def add_parameter(self, name, param_type, bounds, description=""):
           """Add a parameter to the optimization space."""
           self.parameters[name] = {
               'type': param_type,  # 'continuous', 'discrete', 'categorical'
               'bounds': bounds,
               'description': description
           }

       def add_constraint(self, name, constraint_function, description=""):
           """Add a constraint between parameters."""
           self.constraints[name] = {
               'function': constraint_function,
               'description': description
           }

       def sample_parameters(self, n_samples=1):
           """Sample parameters from the defined space."""
           samples = []

           for _ in range(n_samples):
               sample = {}
               for param_name, param_info in self.parameters.items():
                   if param_info['type'] == 'continuous':
                       low, high = param_info['bounds']
                       sample[param_name] = np.random.uniform(low, high)
                   elif param_info['type'] == 'discrete':
                       low, high = param_info['bounds']
                       sample[param_name] = np.random.randint(low, high + 1)
                   elif param_info['type'] == 'categorical':
                       sample[param_name] = np.random.choice(param_info['bounds'])

               # Check constraints
               if self.satisfies_constraints(sample):
                   samples.append(sample)

           return samples

       def satisfies_constraints(self, parameters):
           """Check if parameter set satisfies all constraints."""
           for constraint_name, constraint_info in self.constraints.items():
               if not constraint_info['function'](parameters):
                   return False
           return True

   # Example parameter space for cellular sorting optimization
   def create_sorting_parameter_space():
       space = ParameterSpace()

       # Cellular parameters
       space.add_parameter('population_size', 'discrete', [100, 1000],
                          "Number of cells in simulation")
       space.add_parameter('sorting_strength', 'continuous', [0.1, 2.0],
                          "Strength of cell-cell adhesion")
       space.add_parameter('movement_speed', 'continuous', [0.1, 5.0],
                          "Maximum cell movement speed")
       space.add_parameter('perception_radius', 'continuous', [1.0, 10.0],
                          "Radius for neighbor detection")

       # Simulation parameters
       space.add_parameter('grid_density', 'continuous', [0.1, 0.8],
                          "Fraction of grid occupied by cells")
       space.add_parameter('simulation_time', 'discrete', [1000, 5000],
                          "Number of simulation timesteps")

       # Add constraint: perception radius should not exceed movement speed * 3
       space.add_constraint('perception_movement_constraint',
                           lambda p: p['perception_radius'] <= p['movement_speed'] * 3,
                           "Perception radius should be reasonable relative to movement")

       return space

Step 2: Genetic Algorithm Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement a genetic algorithm for parameter optimization.

.. code-block:: python

   class GeneticOptimizer:
       def __init__(self, parameter_space, fitness_function, population_size=50):
           self.parameter_space = parameter_space
           self.fitness_function = fitness_function
           self.population_size = population_size
           self.mutation_rate = 0.1
           self.crossover_rate = 0.8

       async def optimize(self, generations=20):
           """Run genetic algorithm optimization."""
           print(f"Starting genetic optimization for {generations} generations...")

           # Initialize population
           population = self.parameter_space.sample_parameters(self.population_size)
           fitness_history = []
           best_params_history = []

           for generation in range(generations):
               print(f"Generation {generation + 1}/{generations}")

               # Evaluate fitness
               fitness_scores = []
               for individual in population:
                   fitness = await self.fitness_function(individual)
                   fitness_scores.append(fitness)

               # Track best individual
               best_idx = np.argmax(fitness_scores)
               best_fitness = fitness_scores[best_idx]
               best_params = population[best_idx].copy()

               fitness_history.append(best_fitness)
               best_params_history.append(best_params)

               print(f"  Best fitness: {best_fitness:.4f}")
               print(f"  Best params: {best_params}")

               # Selection, crossover, and mutation
               population = await self.evolve_population(population, fitness_scores)

           return {
               'best_parameters': best_params_history[-1],
               'best_fitness': fitness_history[-1],
               'fitness_history': fitness_history,
               'params_history': best_params_history
           }

       async def evolve_population(self, population, fitness_scores):
           """Evolve population through selection, crossover, and mutation."""
           new_population = []

           # Elitism: keep best individuals
           elite_count = max(1, self.population_size // 10)
           elite_indices = np.argsort(fitness_scores)[-elite_count:]
           for idx in elite_indices:
               new_population.append(population[idx].copy())

           # Generate rest of population through crossover and mutation
           while len(new_population) < self.population_size:
               # Selection
               parent1 = self.tournament_selection(population, fitness_scores)
               parent2 = self.tournament_selection(population, fitness_scores)

               # Crossover
               if np.random.random() < self.crossover_rate:
                   child1, child2 = self.crossover(parent1, parent2)
               else:
                   child1, child2 = parent1.copy(), parent2.copy()

               # Mutation
               if np.random.random() < self.mutation_rate:
                   child1 = self.mutate(child1)
               if np.random.random() < self.mutation_rate:
                   child2 = self.mutate(child2)

               # Add to new population if constraints satisfied
               if self.parameter_space.satisfies_constraints(child1):
                   new_population.append(child1)
               if len(new_population) < self.population_size and \
                  self.parameter_space.satisfies_constraints(child2):
                   new_population.append(child2)

           return new_population[:self.population_size]

       def tournament_selection(self, population, fitness_scores, tournament_size=3):
           """Select individual using tournament selection."""
           tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
           tournament_fitness = [fitness_scores[i] for i in tournament_indices]
           winner_idx = tournament_indices[np.argmax(tournament_fitness)]
           return population[winner_idx].copy()

       def crossover(self, parent1, parent2):
           """Perform crossover between two parents."""
           child1, child2 = {}, {}

           for param_name in parent1.keys():
               if np.random.random() < 0.5:
                   child1[param_name] = parent1[param_name]
                   child2[param_name] = parent2[param_name]
               else:
                   child1[param_name] = parent2[param_name]
                   child2[param_name] = parent1[param_name]

           return child1, child2

       def mutate(self, individual):
           """Mutate an individual."""
           mutated = individual.copy()

           for param_name, param_info in self.parameter_space.parameters.items():
               if np.random.random() < 0.3:  # Mutation probability per parameter
                   if param_info['type'] == 'continuous':
                       low, high = param_info['bounds']
                       # Gaussian mutation
                       mutation = np.random.normal(0, (high - low) * 0.1)
                       mutated[param_name] = np.clip(
                           mutated[param_name] + mutation, low, high
                       )
                   elif param_info['type'] == 'discrete':
                       low, high = param_info['bounds']
                       mutated[param_name] = np.random.randint(low, high + 1)
                   elif param_info['type'] == 'categorical':
                       mutated[param_name] = np.random.choice(param_info['bounds'])

           return mutated

Step 3: Fitness Function Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design fitness functions that evaluate simulation quality.

.. code-block:: python

   class SortingFitnessEvaluator:
       def __init__(self, target_metrics):
           self.target_metrics = target_metrics

       async def evaluate_fitness(self, parameters):
           """Evaluate fitness of parameter set by running simulation."""
           # Run simulation with given parameters
           results = await self.run_simulation(parameters)

           # Calculate fitness based on multiple objectives
           fitness_components = {}

           # Sorting quality component
           sorting_quality = results.get('final_sorting_quality', 0.0)
           target_sorting = self.target_metrics.get('sorting_quality', 0.8)
           fitness_components['sorting'] = 1.0 - abs(sorting_quality - target_sorting)

           # Convergence speed component
           convergence_time = results.get('convergence_timestep', float('inf'))
           max_time = parameters['simulation_time']
           fitness_components['speed'] = max(0.0, 1.0 - convergence_time / max_time)

           # Stability component
           final_energy = results.get('final_energy', 0.0)
           fitness_components['stability'] = max(0.0, 1.0 / (1.0 + final_energy))

           # Resource efficiency component
           efficiency = results.get('computational_efficiency', 0.5)
           fitness_components['efficiency'] = efficiency

           # Combine components with weights
           weights = {
               'sorting': 0.4,
               'speed': 0.3,
               'stability': 0.2,
               'efficiency': 0.1
           }

           total_fitness = sum(weights[component] * value
                              for component, value in fitness_components.items())

           return total_fitness

       async def run_simulation(self, parameters):
           """Run morphogenesis simulation with given parameters."""
           # Initialize simulation
           coordinator = DeterministicCoordinator(
               grid_size=(50, 50),
               max_agents=parameters['population_size']
           )

           # Create agents with specified parameters
           agents = []
           for i in range(parameters['population_size']):
               agent = OptimizedSortingCell(
                   agent_id=i,
                   initial_position=self.generate_random_position(),
                   sorting_strength=parameters['sorting_strength'],
                   movement_speed=parameters['movement_speed'],
                   perception_radius=parameters['perception_radius']
               )
               agents.append(agent)

           await coordinator.add_agents(agents)

           # Run simulation
           start_time = time.time()
           convergence_timestep = None

           for timestep in range(parameters['simulation_time']):
               await coordinator.step()

               # Check convergence periodically
               if timestep % 100 == 0:
                   sorting_quality = await self.calculate_sorting_quality(agents)
                   if sorting_quality > 0.95 and convergence_timestep is None:
                       convergence_timestep = timestep

           end_time = time.time()

           # Calculate final metrics
           final_sorting_quality = await self.calculate_sorting_quality(agents)
           final_energy = await self.calculate_system_energy(agents)
           computational_time = end_time - start_time

           return {
               'final_sorting_quality': final_sorting_quality,
               'convergence_timestep': convergence_timestep or parameters['simulation_time'],
               'final_energy': final_energy,
               'computational_efficiency': 1.0 / max(computational_time, 0.1)
           }

**Exercise:** Create your own fitness function that optimizes for pattern formation quality.

Step 4: Running Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Put everything together to run parameter optimization.

.. code-block:: python

   async def run_optimization_example():
       """Complete optimization example."""
       print("Starting parameter optimization example...")

       # Create parameter space
       param_space = create_sorting_parameter_space()

       # Create fitness evaluator
       target_metrics = {
           'sorting_quality': 0.9,
           'convergence_speed': 0.8,
           'stability': 0.7
       }
       fitness_evaluator = SortingFitnessEvaluator(target_metrics)

       # Run genetic algorithm optimization
       optimizer = GeneticOptimizer(
           parameter_space=param_space,
           fitness_function=fitness_evaluator.evaluate_fitness,
           population_size=20
       )

       results = await optimizer.optimize(generations=10)

       print("\nOptimization Results:")
       print(f"Best fitness: {results['best_fitness']:.4f}")
       print(f"Best parameters: {results['best_parameters']}")

       # Plot optimization progress
       import matplotlib.pyplot as plt

       plt.figure(figsize=(12, 8))

       # Fitness evolution
       plt.subplot(2, 2, 1)
       plt.plot(results['fitness_history'])
       plt.title('Best Fitness Over Generations')
       plt.xlabel('Generation')
       plt.ylabel('Fitness')

       # Parameter evolution
       param_names = list(results['best_parameters'].keys())

       for i, param_name in enumerate(param_names[:4]):  # Show first 4 parameters
           plt.subplot(2, 2, i + 2) if i < 3 else plt.subplot(2, 2, 4)
           param_values = [params[param_name] for params in results['params_history']]
           plt.plot(param_values)
           plt.title(f'{param_name} Evolution')
           plt.xlabel('Generation')
           plt.ylabel(param_name)

       plt.tight_layout()
       plt.savefig('optimization_results.png', dpi=300)
       plt.show()

       return results

   # Run the optimization
   if __name__ == "__main__":
       optimization_results = asyncio.run(run_optimization_example())

**Exercise:** Implement Bayesian optimization as an alternative to genetic algorithms.

Tutorial 3: Advanced Data Analysis Workflows
---------------------------------------------

Learn how to analyze morphogenesis simulation results using statistical methods and machine learning techniques.

**Learning Objectives:**
- Perform statistical analysis of simulation results
- Use machine learning for pattern recognition
- Create publication-quality visualizations
- Implement hypothesis testing workflows

Step 1: Data Collection Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a comprehensive data collection system for simulations.

.. code-block:: python

   import pandas as pd
   from pathlib import Path
   import pickle

   class SimulationDataCollector:
       def __init__(self, experiment_name):
           self.experiment_name = experiment_name
           self.data_storage = {
               'temporal_data': [],
               'spatial_data': [],
               'agent_data': [],
               'system_metrics': [],
               'metadata': {}
           }

           # Create output directory
           self.output_dir = Path(f"data/{experiment_name}")
           self.output_dir.mkdir(parents=True, exist_ok=True)

       async def collect_timestep_data(self, timestep, agents, system_state):
           """Collect data for a single timestep."""
           # Temporal data
           temporal_entry = {
               'timestep': timestep,
               'timestamp': time.time(),
               'num_agents': len(agents),
               'system_energy': await self.calculate_system_energy(system_state),
               'order_parameter': await self.calculate_order_parameter(agents)
           }
           self.data_storage['temporal_data'].append(temporal_entry)

           # Spatial data
           spatial_entry = {
               'timestep': timestep,
               'positions': [(a.position.x, a.position.y) for a in agents],
               'cell_types': [getattr(a, 'cell_type', 'default') for a in agents],
               'local_densities': [await a.calculate_local_density() for a in agents]
           }
           self.data_storage['spatial_data'].append(spatial_entry)

           # Individual agent data (sampled for efficiency)
           if timestep % 10 == 0:  # Sample every 10 timesteps
               agent_entries = []
               for agent in agents[::5]:  # Sample every 5th agent
                   agent_entry = {
                       'timestep': timestep,
                       'agent_id': agent.agent_id,
                       'position_x': agent.position.x,
                       'position_y': agent.position.y,
                       'energy': getattr(agent, 'energy', 0.0),
                       'age': getattr(agent, 'age', 0),
                       'local_neighbors': len(await agent.get_neighbors())
                   }
                   agent_entries.append(agent_entry)

               self.data_storage['agent_data'].extend(agent_entries)

       def save_data(self):
           """Save collected data to files."""
           # Save as pandas DataFrames
           temporal_df = pd.DataFrame(self.data_storage['temporal_data'])
           temporal_df.to_csv(self.output_dir / 'temporal_data.csv', index=False)

           spatial_df = pd.DataFrame(self.data_storage['spatial_data'])
           spatial_df.to_pickle(self.output_dir / 'spatial_data.pkl')

           agent_df = pd.DataFrame(self.data_storage['agent_data'])
           agent_df.to_csv(self.output_dir / 'agent_data.csv', index=False)

           # Save metadata
           with open(self.output_dir / 'metadata.pkl', 'wb') as f:
               pickle.dump(self.data_storage['metadata'], f)

           print(f"Data saved to {self.output_dir}")

Step 2: Statistical Analysis Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement statistical analysis methods for morphogenesis data.

.. code-block:: python

   from scipy import stats
   import seaborn as sns

   class MorphogenesisAnalyzer:
       def __init__(self, data_path):
           self.data_path = Path(data_path)
           self.temporal_data = pd.read_csv(self.data_path / 'temporal_data.csv')
           self.agent_data = pd.read_csv(self.data_path / 'agent_data.csv')

           with open(self.data_path / 'spatial_data.pkl', 'rb') as f:
               self.spatial_data = pd.read_pickle(f)

       def analyze_convergence(self):
           """Analyze convergence properties."""
           convergence_results = {}

           # Find convergence point
           order_params = self.temporal_data['order_parameter'].values
           convergence_threshold = 0.95 * order_params[-1]  # 95% of final value

           convergence_idx = np.argmax(order_params >= convergence_threshold)
           convergence_time = self.temporal_data.iloc[convergence_idx]['timestep']

           convergence_results['convergence_time'] = convergence_time
           convergence_results['final_order'] = order_params[-1]
           convergence_results['convergence_rate'] = self.calculate_convergence_rate()

           return convergence_results

       def calculate_convergence_rate(self):
           """Calculate exponential convergence rate."""
           order_params = self.temporal_data['order_parameter'].values
           timesteps = self.temporal_data['timestep'].values

           # Fit exponential: order(t) = A * (1 - exp(-k*t))
           def exponential_convergence(t, A, k):
               return A * (1 - np.exp(-k * t))

           try:
               popt, _ = stats.curve_fit(
                   exponential_convergence,
                   timesteps,
                   order_params,
                   p0=[1.0, 0.001]
               )
               return popt[1]  # Convergence rate k
           except:
               return np.nan

       def analyze_spatial_patterns(self):
           """Analyze spatial organization patterns."""
           pattern_results = {}

           # Analyze final spatial configuration
           final_spatial = self.spatial_data.iloc[-1]
           positions = np.array(final_spatial['positions'])
           cell_types = final_spatial['cell_types']

           # Calculate spatial clustering
           pattern_results['spatial_clustering'] = self.calculate_spatial_clustering(
               positions, cell_types
           )

           # Calculate pattern regularity
           pattern_results['pattern_regularity'] = self.calculate_pattern_regularity(positions)

           # Analyze boundary sharpness
           pattern_results['boundary_sharpness'] = self.calculate_boundary_sharpness(
               positions, cell_types
           )

           return pattern_results

       def calculate_spatial_clustering(self, positions, cell_types):
           """Calculate degree of spatial clustering by cell type."""
           clustering_scores = {}

           unique_types = np.unique(cell_types)
           for cell_type in unique_types:
               type_positions = positions[np.array(cell_types) == cell_type]

               if len(type_positions) < 2:
                   clustering_scores[cell_type] = 0.0
                   continue

               # Calculate nearest neighbor distances within type
               intra_distances = []
               for i, pos in enumerate(type_positions):
                   other_positions = np.delete(type_positions, i, axis=0)
                   distances = np.linalg.norm(other_positions - pos, axis=1)
                   intra_distances.append(np.min(distances))

               # Calculate nearest neighbor distances to other types
               other_positions = positions[np.array(cell_types) != cell_type]
               inter_distances = []
               for pos in type_positions:
                   if len(other_positions) > 0:
                       distances = np.linalg.norm(other_positions - pos, axis=1)
                       inter_distances.append(np.min(distances))

               # Clustering score: ratio of inter to intra distances
               if intra_distances and inter_distances:
                   clustering_scores[cell_type] = (
                       np.mean(inter_distances) / np.mean(intra_distances)
                   )
               else:
                   clustering_scores[cell_type] = 1.0

           return clustering_scores

       def perform_hypothesis_testing(self, experimental_conditions):
           """Perform statistical hypothesis testing between conditions."""
           test_results = {}

           # Compare convergence times between conditions
           convergence_times = []
           condition_labels = []

           for condition, data_path in experimental_conditions.items():
               condition_analyzer = MorphogenesisAnalyzer(data_path)
               convergence_result = condition_analyzer.analyze_convergence()
               convergence_times.append(convergence_result['convergence_time'])
               condition_labels.append(condition)

           # Perform ANOVA if multiple conditions
           if len(experimental_conditions) > 2:
               f_stat, p_value = stats.f_oneway(*[
                   [time] for time in convergence_times
               ])
               test_results['anova'] = {'f_statistic': f_stat, 'p_value': p_value}

           # Pairwise t-tests
           if len(experimental_conditions) == 2:
               t_stat, p_value = stats.ttest_ind(
                   convergence_times[0], convergence_times[1]
               )
               test_results['t_test'] = {'t_statistic': t_stat, 'p_value': p_value}

           return test_results

       def create_analysis_report(self):
           """Generate comprehensive analysis report."""
           report = {}

           # Basic statistics
           report['basic_stats'] = {
               'total_timesteps': len(self.temporal_data),
               'final_num_agents': self.temporal_data.iloc[-1]['num_agents'],
               'simulation_duration': self.temporal_data.iloc[-1]['timestep']
           }

           # Convergence analysis
           report['convergence'] = self.analyze_convergence()

           # Spatial pattern analysis
           report['spatial_patterns'] = self.analyze_spatial_patterns()

           # Time series statistics
           report['temporal_stats'] = {
               'mean_order_parameter': self.temporal_data['order_parameter'].mean(),
               'std_order_parameter': self.temporal_data['order_parameter'].std(),
               'mean_system_energy': self.temporal_data['system_energy'].mean()
           }

           return report

Step 3: Visualization Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create advanced visualization tools for analysis results.

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   from matplotlib.animation import FuncAnimation

   class MorphogenesisVisualizer:
       def __init__(self, analyzer):
           self.analyzer = analyzer
           self.setup_style()

       def setup_style(self):
           """Setup publication-quality plot style."""
           plt.style.use('seaborn-v0_8-whitegrid')
           sns.set_palette("husl")

           # Custom parameters for publication
           plt.rcParams.update({
               'font.size': 12,
               'axes.titlesize': 14,
               'axes.labelsize': 12,
               'xtick.labelsize': 10,
               'ytick.labelsize': 10,
               'legend.fontsize': 10,
               'figure.titlesize': 16
           })

       def plot_convergence_analysis(self):
           """Create convergence analysis plots."""
           fig, axes = plt.subplots(2, 2, figsize=(15, 10))

           # Order parameter over time
           axes[0, 0].plot(self.analyzer.temporal_data['timestep'],
                          self.analyzer.temporal_data['order_parameter'])
           axes[0, 0].set_title('Order Parameter Evolution')
           axes[0, 0].set_xlabel('Timestep')
           axes[0, 0].set_ylabel('Order Parameter')

           # System energy over time
           axes[0, 1].plot(self.analyzer.temporal_data['timestep'],
                          self.analyzer.temporal_data['system_energy'])
           axes[0, 1].set_title('System Energy Evolution')
           axes[0, 1].set_xlabel('Timestep')
           axes[0, 1].set_ylabel('System Energy')

           # Agent count over time
           axes[1, 0].plot(self.analyzer.temporal_data['timestep'],
                          self.analyzer.temporal_data['num_agents'])
           axes[1, 0].set_title('Agent Count Over Time')
           axes[1, 0].set_xlabel('Timestep')
           axes[1, 0].set_ylabel('Number of Agents')

           # Convergence rate analysis
           convergence_analysis = self.analyzer.analyze_convergence()
           axes[1, 1].text(0.1, 0.8, f"Convergence Time: {convergence_analysis['convergence_time']}")
           axes[1, 1].text(0.1, 0.6, f"Final Order: {convergence_analysis['final_order']:.3f}")
           axes[1, 1].text(0.1, 0.4, f"Convergence Rate: {convergence_analysis['convergence_rate']:.6f}")
           axes[1, 1].set_title('Convergence Statistics')
           axes[1, 1].axis('off')

           plt.tight_layout()
           return fig

       def plot_spatial_analysis(self):
           """Create spatial pattern analysis plots."""
           spatial_analysis = self.analyzer.analyze_spatial_patterns()

           fig, axes = plt.subplots(2, 2, figsize=(15, 10))

           # Final spatial configuration
           final_spatial = self.analyzer.spatial_data.iloc[-1]
           positions = np.array(final_spatial['positions'])
           cell_types = final_spatial['cell_types']

           unique_types = np.unique(cell_types)
           colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))

           for i, cell_type in enumerate(unique_types):
               type_positions = positions[np.array(cell_types) == cell_type]
               if len(type_positions) > 0:
                   axes[0, 0].scatter(type_positions[:, 0], type_positions[:, 1],
                                    c=[colors[i]], label=cell_type, alpha=0.7)

           axes[0, 0].set_title('Final Spatial Configuration')
           axes[0, 0].legend()
           axes[0, 0].set_xlabel('X Position')
           axes[0, 0].set_ylabel('Y Position')

           # Spatial clustering by type
           clustering_scores = spatial_analysis['spatial_clustering']
           axes[0, 1].bar(clustering_scores.keys(), clustering_scores.values())
           axes[0, 1].set_title('Spatial Clustering by Cell Type')
           axes[0, 1].set_ylabel('Clustering Score')

           # Pattern regularity over time
           # This would require storing pattern regularity at each timestep
           axes[1, 0].text(0.5, 0.5, 'Pattern Regularity Analysis\n(Implementation needed)',
                          ha='center', va='center', transform=axes[1, 0].transAxes)
           axes[1, 0].set_title('Pattern Regularity Over Time')

           # Boundary sharpness analysis
           axes[1, 1].text(0.5, 0.5, f'Boundary Sharpness: {spatial_analysis["boundary_sharpness"]:.3f}',
                          ha='center', va='center', transform=axes[1, 1].transAxes)
           axes[1, 1].set_title('Boundary Sharpness')
           axes[1, 1].axis('off')

           plt.tight_layout()
           return fig

**Exercise:** Implement pattern regularity calculation and visualization.

Step 4: Complete Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Put everything together in a complete analysis workflow.

.. code-block:: python

   async def run_analysis_workflow():
       """Complete data analysis workflow example."""
       print("Starting comprehensive data analysis workflow...")

       # Simulate data collection (normally this would be from actual simulations)
       collector = SimulationDataCollector("analysis_example")

       # Load and analyze data
       analyzer = MorphogenesisAnalyzer("data/analysis_example")

       # Generate analysis report
       report = analyzer.create_analysis_report()
       print("\nAnalysis Report:")
       print(f"Convergence time: {report['convergence']['convergence_time']}")
       print(f"Final order parameter: {report['convergence']['final_order']:.3f}")
       print(f"Convergence rate: {report['convergence']['convergence_rate']:.6f}")

       # Create visualizations
       visualizer = MorphogenesisVisualizer(analyzer)

       # Convergence plots
       conv_fig = visualizer.plot_convergence_analysis()
       conv_fig.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')

       # Spatial analysis plots
       spatial_fig = visualizer.plot_spatial_analysis()
       spatial_fig.savefig('spatial_analysis.png', dpi=300, bbox_inches='tight')

       plt.show()

       return report

   # Run the complete workflow
   if __name__ == "__main__":
       analysis_report = asyncio.run(run_analysis_workflow())

**Exercise:** Extend the analysis to include machine learning-based pattern classification.

Summary
-------

These intermediate tutorials have covered:

1. **Multi-Scale Modeling:**
   - Gene regulatory networks
   - Scale bridges and information transfer
   - Cross-scale coupling mechanisms

2. **Optimization Techniques:**
   - Parameter space definition
   - Genetic algorithm implementation
   - Fitness function design

3. **Advanced Data Analysis:**
   - Comprehensive data collection
   - Statistical analysis methods
   - Publication-quality visualization

**Next Steps:**
- Practice implementing your own multi-scale models
- Experiment with different optimization algorithms
- Develop custom analysis tools for your specific research questions

**Additional Resources:**
- Advanced tutorials section for more complex topics
- Research tutorials for cutting-edge applications
- Technical tutorials for performance optimization