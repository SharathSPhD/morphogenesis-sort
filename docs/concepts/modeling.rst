Computational Modeling
=====================

This section explores the computational modeling approaches used in morphogenesis research, covering different modeling paradigms, implementation strategies, and validation techniques for creating realistic simulations of biological development processes.

Introduction to Computational Modeling
--------------------------------------

Computational modeling in morphogenesis involves creating mathematical and algorithmic representations of biological processes that can be simulated on computers to understand, predict, and control developmental outcomes.

**Key Modeling Approaches:**

* **Agent-Based Models**: Individual cells as autonomous agents
* **Continuous Field Models**: Morphogen concentrations and gradients
* **Hybrid Models**: Combining discrete agents with continuous fields
* **Multi-Scale Models**: Linking molecular, cellular, and tissue scales
* **Stochastic Models**: Incorporating biological noise and variability

**Model Selection Criteria:**

* **Spatial Scale**: From molecular to tissue level
* **Temporal Scale**: From milliseconds to days/years
* **Biological Complexity**: Single cell type vs. multiple interacting types
* **Computational Resources**: Available memory and processing power
* **Validation Requirements**: Experimental data availability

Agent-Based Modeling
--------------------

Agent-based models (ABMs) represent individual cells as autonomous computational agents that follow behavioral rules and interact with their environment.

Core Agent Architecture
~~~~~~~~~~~~~~~~~~~~~~~

**Basic Agent Structure:**

.. code-block:: python

   class MorphogeneticAgent:
       def __init__(self, agent_id, initial_position, cell_type):
           self.agent_id = agent_id
           self.position = initial_position
           self.cell_type = cell_type
           self.internal_state = {}
           self.behavioral_rules = []

       async def sense(self):
           """Gather information about local environment."""
           perception = {
               'neighbors': await self.detect_neighbors(),
               'morphogens': await self.sense_morphogens(),
               'mechanical_forces': await self.sense_forces(),
               'substrate_properties': await self.sense_substrate()
           }
           return perception

       async def decide(self, perception):
           """Make decisions based on perceived information."""
           decision = {'actions': []}

           for rule in self.behavioral_rules:
               action = await rule.evaluate(perception, self.internal_state)
               if action:
                   decision['actions'].append(action)

           return decision

       async def act(self, decision):
           """Execute decided actions."""
           for action in decision['actions']:
               await self.execute_action(action)

       async def update_state(self):
           """Update internal state after actions."""
           await self.update_biochemistry()
           await self.update_mechanical_properties()
           await self.update_age_and_lifecycle()

**Behavioral Rule Framework:**

.. code-block:: python

   class BehavioralRule:
       def __init__(self, name, condition, action, priority=1.0):
           self.name = name
           self.condition = condition
           self.action = action
           self.priority = priority

       async def evaluate(self, perception, internal_state):
           """Evaluate if rule should fire."""
           if await self.condition(perception, internal_state):
               return await self.action(perception, internal_state)
           return None

   # Example behavioral rules
   class ChemotaxisRule(BehavioralRule):
       def __init__(self, target_morphogen, sensitivity=1.0):
           self.target_morphogen = target_morphogen
           self.sensitivity = sensitivity

           async def condition(perception, state):
               return self.target_morphogen in perception['morphogens']

           async def action(perception, state):
               gradient = perception['morphogens'][self.target_morphogen]['gradient']
               move_direction = gradient * self.sensitivity
               return {'type': 'move', 'direction': move_direction}

           super().__init__('chemotaxis', condition, action)

   class DivisionRule(BehavioralRule):
       def __init__(self, division_threshold=0.8, cell_cycle_duration=24):
           self.threshold = division_threshold
           self.cycle_duration = cell_cycle_duration

           async def condition(perception, state):
               return (state.get('cell_cycle_progress', 0) >= self.threshold and
                       state.get('division_competent', True))

           async def action(perception, state):
               return {'type': 'divide', 'axis': await self.calculate_division_axis(perception)}

           super().__init__('division', condition, action)

Multi-Scale Agent Models
~~~~~~~~~~~~~~~~~~~~~~~~

**Hierarchical Agent Organization:**

.. code-block:: python

   class MultiScaleAgentSystem:
       def __init__(self):
           self.molecular_agents = []    # Protein/gene networks
           self.cellular_agents = []     # Individual cells
           self.tissue_agents = []       # Cell groups/tissues

       async def update_molecular_scale(self, dt):
           """Update molecular-level processes."""
           for molecule in self.molecular_agents:
               await molecule.update_concentration(dt)
               await molecule.process_reactions(dt)

       async def update_cellular_scale(self, dt):
           """Update cellular-level processes."""
           # Cellular agents read molecular states
           for cell in self.cellular_agents:
               molecular_context = await self.get_molecular_context(cell)
               await cell.update_from_molecular_state(molecular_context)
               await cell.step(dt)

           # Update intercellular interactions
           await self.process_cell_interactions()

       async def update_tissue_scale(self, dt):
           """Update tissue-level processes."""
           for tissue in self.tissue_agents:
               cellular_context = await self.get_cellular_context(tissue)
               await tissue.update_from_cellular_state(cellular_context)
               await tissue.apply_tissue_level_constraints()

       async def cross_scale_communication(self):
           """Handle information flow between scales."""
           # Bottom-up: molecular -> cellular -> tissue
           await self.molecular_to_cellular_signaling()
           await self.cellular_to_tissue_signaling()

           # Top-down: tissue -> cellular -> molecular
           await self.tissue_to_cellular_signaling()
           await self.cellular_to_molecular_signaling()

**Implementation Example:**

.. code-block:: python

   class MultiScaleCell(MorphogeneticAgent):
       def __init__(self, agent_id, position, cell_type):
           super().__init__(agent_id, position, cell_type)
           self.gene_network = GeneRegulatoryNetwork()
           self.protein_network = ProteinInteractionNetwork()
           self.mechanical_state = MechanicalState()

       async def update_molecular_level(self, dt):
           """Update gene and protein networks."""
           # Gene expression
           gene_expression = await self.gene_network.update(
               external_signals=self.get_external_signals(),
               dt=dt
           )

           # Protein interactions
           protein_levels = await self.protein_network.update(
               gene_expression=gene_expression,
               dt=dt
           )

           # Update cellular properties based on molecular state
           await self.update_cellular_properties(protein_levels)

       async def update_cellular_level(self, dt):
           """Update cellular behavior and interactions."""
           perception = await self.sense()
           decision = await self.decide(perception)
           await self.act(decision)

       async def update_mechanical_level(self, dt):
           """Update mechanical properties and forces."""
           forces = await self.calculate_forces()
           await self.mechanical_state.update(forces, dt)

           # Mechanical feedback to cellular behavior
           if self.mechanical_state.stress > self.stress_threshold:
               await self.trigger_stress_response()

Continuous Field Modeling
-------------------------

Continuous field models represent morphogens, mechanical stresses, and other physical quantities as continuous functions over space and time.

Reaction-Diffusion Models
~~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-Species Reaction-Diffusion:**

.. code-block:: python

   class ReactionDiffusionModel:
       def __init__(self, species_names, domain_size, dx=0.1, dt=0.001):
           self.species = species_names
           self.n_species = len(species_names)
           self.domain_size = domain_size
           self.dx = dx
           self.dt = dt

           # Initialize concentration fields
           self.concentrations = {}
           for species in species_names:
               self.concentrations[species] = np.zeros(domain_size)

           # Model parameters
           self.diffusion_coefficients = {}
           self.reaction_terms = {}

       def add_reaction_term(self, species, reaction_function):
           """Add reaction term for a species."""
           self.reaction_terms[species] = reaction_function

       def set_diffusion_coefficient(self, species, coefficient):
           """Set diffusion coefficient for a species."""
           self.diffusion_coefficients[species] = coefficient

       def laplacian(self, field):
           """Calculate Laplacian using finite differences."""
           if len(field.shape) == 1:  # 1D
               laplacian = np.zeros_like(field)
               laplacian[1:-1] = (field[2:] - 2*field[1:-1] + field[:-2]) / self.dx**2
           elif len(field.shape) == 2:  # 2D
               laplacian = np.zeros_like(field)
               laplacian[1:-1, 1:-1] = (
                   (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / self.dx**2 +
                   (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / self.dx**2
               )
           return laplacian

       async def step(self):
           """Single time step of the reaction-diffusion system."""
           new_concentrations = {}

           for species in self.species:
               current_conc = self.concentrations[species]

               # Diffusion term
               diffusion = self.diffusion_coefficients.get(species, 0) * self.laplacian(current_conc)

               # Reaction term
               if species in self.reaction_terms:
                   reaction = await self.reaction_terms[species](self.concentrations)
               else:
                   reaction = 0

               # Update concentration
               new_concentrations[species] = current_conc + self.dt * (diffusion + reaction)

           self.concentrations = new_concentrations

       def set_boundary_conditions(self, species, boundary_type, boundary_value):
           """Set boundary conditions for a species."""
           if boundary_type == 'dirichlet':
               # Fixed concentration at boundaries
               self.concentrations[species][0] = boundary_value
               self.concentrations[species][-1] = boundary_value
           elif boundary_type == 'neumann':
               # Fixed gradient at boundaries (zero flux)
               self.concentrations[species][0] = self.concentrations[species][1]
               self.concentrations[species][-1] = self.concentrations[species][-2]

**Complex Reaction Networks:**

.. code-block:: python

   class MorphogenNetwork:
       def __init__(self):
           self.reactions = []

       def add_production_reaction(self, source_species, target_species, rate):
           """Add production reaction: source -> target."""
           def reaction(concentrations):
               return rate * concentrations[source_species]

           self.reactions.append({
               'target': target_species,
               'function': reaction,
               'type': 'production'
           })

       def add_degradation_reaction(self, species, rate):
           """Add degradation reaction: species -> ∅."""
           def reaction(concentrations):
               return -rate * concentrations[species]

           self.reactions.append({
               'target': species,
               'function': reaction,
               'type': 'degradation'
           })

       def add_michaelis_menten_reaction(self, substrate, enzyme, product, vmax, km):
           """Add Michaelis-Menten reaction: substrate + enzyme -> product + enzyme."""
           def reaction_substrate(concentrations):
               s = concentrations[substrate]
               e = concentrations[enzyme]
               return -vmax * e * s / (km + s)

           def reaction_product(concentrations):
               s = concentrations[substrate]
               e = concentrations[enzyme]
               return vmax * e * s / (km + s)

           self.reactions.extend([
               {'target': substrate, 'function': reaction_substrate, 'type': 'consumption'},
               {'target': product, 'function': reaction_product, 'type': 'production'}
           ])

       async def calculate_reaction_rates(self, concentrations):
           """Calculate all reaction rates."""
           reaction_rates = {species: 0 for species in concentrations.keys()}

           for reaction in self.reactions:
               target_species = reaction['target']
               rate = await reaction['function'](concentrations)
               reaction_rates[target_species] += rate

           return reaction_rates

Mechanical Modeling
~~~~~~~~~~~~~~~~~~~

**Cellular Mechanics with Vertex Models:**

.. code-block:: python

   class VertexModel:
       def __init__(self, vertices, cells):
           self.vertices = vertices  # List of vertex positions
           self.cells = cells        # List of cells (vertex indices)
           self.forces = np.zeros_like(vertices)

       def calculate_cell_area(self, cell_vertices):
           """Calculate area of a cell using shoelace formula."""
           x = cell_vertices[:, 0]
           y = cell_vertices[:, 1]
           return 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

       def calculate_cell_perimeter(self, cell_vertices):
           """Calculate perimeter of a cell."""
           perimeter = 0
           for i in range(len(cell_vertices)):
               j = (i + 1) % len(cell_vertices)
               edge_length = np.linalg.norm(cell_vertices[j] - cell_vertices[i])
               perimeter += edge_length
           return perimeter

       def calculate_elastic_forces(self):
           """Calculate elastic forces on vertices."""
           self.forces.fill(0)

           for cell_id, vertex_indices in enumerate(self.cells):
               cell_vertices = self.vertices[vertex_indices]

               # Area constraint force
               current_area = self.calculate_cell_area(cell_vertices)
               target_area = self.cell_target_areas[cell_id]
               area_tension = self.area_elasticity * (current_area - target_area)

               # Perimeter constraint force
               current_perimeter = self.calculate_cell_perimeter(cell_vertices)
               perimeter_tension = self.perimeter_elasticity * current_perimeter

               # Calculate forces on each vertex
               for i, vertex_idx in enumerate(vertex_indices):
                   # Area force
                   j_prev = (i - 1) % len(vertex_indices)
                   j_next = (i + 1) % len(vertex_indices)

                   v_prev = cell_vertices[j_prev]
                   v_next = cell_vertices[j_next]

                   area_force = area_tension * np.array([
                       -(v_next[1] - v_prev[1]),
                       v_next[0] - v_prev[0]
                   ]) / 2

                   # Perimeter force
                   edge_vectors = []
                   edge_vectors.append(cell_vertices[i] - v_prev)
                   edge_vectors.append(v_next - cell_vertices[i])

                   perimeter_force = np.zeros(2)
                   for edge_vec in edge_vectors:
                       edge_length = np.linalg.norm(edge_vec)
                       if edge_length > 0:
                           perimeter_force += perimeter_tension * edge_vec / edge_length

                   self.forces[vertex_idx] += area_force + perimeter_force

       async def step(self, dt):
           """Single time step of vertex dynamics."""
           self.calculate_elastic_forces()

           # Add viscous drag
           velocities = -self.forces / self.drag_coefficient

           # Update vertex positions
           self.vertices += velocities * dt

           # Apply constraints (e.g., boundary conditions)
           await self.apply_constraints()

Hybrid Agent-Field Models
-------------------------

Combining discrete cellular agents with continuous morphogen fields.

Agent-Field Coupling
~~~~~~~~~~~~~~~~~~~~

**Bidirectional Coupling:**

.. code-block:: python

   class HybridMorphogeneticSystem:
       def __init__(self, agents, morphogen_fields):
           self.agents = agents
           self.morphogen_fields = morphogen_fields
           self.coupling_strength = 1.0

       async def agent_to_field_coupling(self):
           """Agents influence morphogen fields."""
           for agent in self.agents:
               position = agent.position
               cell_type = agent.cell_type

               # Morphogen production by cells
               for morphogen_name, field in self.morphogen_fields.items():
                   production_rate = agent.get_production_rate(morphogen_name)

                   if production_rate > 0:
                       # Add morphogen at agent position
                       field_indices = self.position_to_field_indices(position)
                       field.add_source(field_indices, production_rate)

               # Morphogen consumption by cells
               for morphogen_name, field in self.morphogen_fields.items():
                   consumption_rate = agent.get_consumption_rate(morphogen_name)

                   if consumption_rate > 0:
                       field_indices = self.position_to_field_indices(position)
                       field.add_sink(field_indices, consumption_rate)

       async def field_to_agent_coupling(self):
           """Morphogen fields influence agent behavior."""
           for agent in self.agents:
               position = agent.position

               # Sample morphogen concentrations at agent position
               local_concentrations = {}
               local_gradients = {}

               for morphogen_name, field in self.morphogen_fields.items():
                   concentration = field.interpolate_at_position(position)
                   gradient = field.calculate_gradient_at_position(position)

                   local_concentrations[morphogen_name] = concentration
                   local_gradients[morphogen_name] = gradient

               # Update agent's morphogen perception
               agent.update_morphogen_perception(local_concentrations, local_gradients)

       async def step(self, dt):
           """Single time step of hybrid system."""
           # Update morphogen fields
           for field in self.morphogen_fields.values():
               await field.step(dt)

           # Agent-field coupling
           await self.agent_to_field_coupling()
           await self.field_to_agent_coupling()

           # Update agents
           for agent in self.agents:
               await agent.step(dt)

**Adaptive Mesh Refinement:**

.. code-block:: python

   class AdaptiveMorphogenField:
       def __init__(self, initial_resolution, refinement_criteria):
           self.resolution = initial_resolution
           self.refinement_criteria = refinement_criteria
           self.mesh = self.create_initial_mesh()

       def create_initial_mesh(self):
           """Create initial uniform mesh."""
           # Implementation details for mesh creation
           pass

       def assess_refinement_needs(self):
           """Determine which regions need mesh refinement."""
           refinement_regions = []

           for cell in self.mesh.cells:
               # Check refinement criteria
               gradient_magnitude = self.calculate_gradient_magnitude(cell)
               agent_density = self.calculate_local_agent_density(cell)

               if (gradient_magnitude > self.refinement_criteria['max_gradient'] or
                   agent_density > self.refinement_criteria['max_agent_density']):
                   refinement_regions.append(cell)

           return refinement_regions

       def refine_mesh(self, regions_to_refine):
           """Refine mesh in specified regions."""
           for region in regions_to_refine:
               self.subdivide_cell(region)

       def coarsen_mesh(self, regions_to_coarsen):
           """Coarsen mesh where high resolution is not needed."""
           for region in regions_to_coarsen:
               self.merge_cells(region)

       async def adaptive_step(self, dt):
           """Step with adaptive mesh refinement."""
           # Standard field update
           await self.step(dt)

           # Assess and perform mesh adaptation
           if self.should_adapt_mesh():
               refinement_regions = self.assess_refinement_needs()
               coarsening_regions = self.assess_coarsening_opportunities()

               self.refine_mesh(refinement_regions)
               self.coarsen_mesh(coarsening_regions)

               # Interpolate field values to new mesh
               await self.interpolate_to_new_mesh()

Stochastic Modeling
-------------------

Incorporating biological noise and variability into morphogenetic models.

Noise Sources in Biological Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Intrinsic Noise:**

.. code-block:: python

   class StochasticAgent(MorphogeneticAgent):
       def __init__(self, agent_id, position, cell_type, noise_level=0.1):
           super().__init__(agent_id, position, cell_type)
           self.noise_level = noise_level
           self.random_generator = np.random.RandomState()

       async def stochastic_gene_expression(self, base_expression_level):
           """Model stochastic gene expression."""
           # Poisson noise in transcription
           mean_transcripts = base_expression_level
           actual_transcripts = self.random_generator.poisson(mean_transcripts)

           # Gamma noise in translation
           shape = actual_transcripts
           scale = 1.0
           protein_level = self.random_generator.gamma(shape, scale) if shape > 0 else 0

           return protein_level

       async def stochastic_decision_making(self, decision_probabilities):
           """Make stochastic decisions based on probabilities."""
           random_value = self.random_generator.random()
           cumulative_prob = 0

           for decision, probability in decision_probabilities.items():
               cumulative_prob += probability
               if random_value <= cumulative_prob:
                   return decision

           return list(decision_probabilities.keys())[-1]  # Fallback

       async def add_movement_noise(self, intended_movement):
           """Add noise to movement decisions."""
           noise_vector = self.random_generator.normal(
               0, self.noise_level, size=intended_movement.shape
           )
           return intended_movement + noise_vector

**Extrinsic Noise:**

.. code-block:: python

   class NoisyEnvironment:
       def __init__(self, temperature_fluctuations=True, mechanical_noise=True):
           self.temperature_fluctuations = temperature_fluctuations
           self.mechanical_noise = mechanical_noise
           self.base_temperature = 37.0  # Celsius
           self.noise_generators = {}

       def add_correlated_noise(self, correlation_time, amplitude):
           """Add temporally correlated noise (Ornstein-Uhlenbeck process)."""
           class OUProcess:
               def __init__(self, tau, sigma):
                   self.tau = tau  # Correlation time
                   self.sigma = sigma  # Noise amplitude
                   self.current_value = 0
                   self.dt = 0.1

               def step(self):
                   # Ornstein-Uhlenbeck update
                   drift = -self.current_value / self.tau
                   diffusion = self.sigma * np.sqrt(2 / self.tau) * np.random.normal()
                   self.current_value += (drift * self.dt + diffusion * np.sqrt(self.dt))
                   return self.current_value

           return OUProcess(correlation_time, amplitude)

       async def apply_environmental_fluctuations(self, agents):
           """Apply environmental noise to agents."""
           if self.temperature_fluctuations:
               temp_noise = self.noise_generators.get('temperature')
               if temp_noise:
                   temperature_deviation = temp_noise.step()
                   current_temp = self.base_temperature + temperature_deviation

                   for agent in agents:
                       await agent.apply_temperature_effect(current_temp)

Gillespie Algorithm for Chemical Reactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Exact Stochastic Simulation:**

.. code-block:: python

   class GillespieSimulator:
       def __init__(self, reactions, initial_state):
           self.reactions = reactions
           self.state = initial_state.copy()
           self.time = 0.0

       def calculate_propensities(self):
           """Calculate reaction propensities (rates)."""
           propensities = []

           for reaction in self.reactions:
               propensity = reaction['rate']

               # Multiply by reactant concentrations
               for reactant, stoichiometry in reaction['reactants'].items():
                   concentration = self.state[reactant]
                   # Binomial coefficient for multiple molecules
                   for k in range(stoichiometry):
                       propensity *= (concentration - k)
                       if propensity <= 0:
                           propensity = 0
                           break

               propensities.append(max(0, propensity))

           return propensities

       async def gillespie_step(self):
           """Single Gillespie algorithm step."""
           propensities = self.calculate_propensities()
           total_propensity = sum(propensities)

           if total_propensity == 0:
               return False  # No more reactions possible

           # Time to next reaction
           tau = -np.log(np.random.random()) / total_propensity
           self.time += tau

           # Select which reaction occurs
           reaction_selector = np.random.random() * total_propensity
           cumulative_propensity = 0

           for i, propensity in enumerate(propensities):
               cumulative_propensity += propensity
               if reaction_selector <= cumulative_propensity:
                   # Execute reaction i
                   await self.execute_reaction(i)
                   break

           return True

       async def execute_reaction(self, reaction_index):
           """Execute a specific reaction."""
           reaction = self.reactions[reaction_index]

           # Consume reactants
           for reactant, stoichiometry in reaction['reactants'].items():
               self.state[reactant] -= stoichiometry

           # Produce products
           for product, stoichiometry in reaction['products'].items():
               self.state[product] = self.state.get(product, 0) + stoichiometry

       async def simulate_until(self, end_time):
           """Simulate until specified time."""
           trajectory = [(self.time, self.state.copy())]

           while self.time < end_time:
               if not await self.gillespie_step():
                   break  # No more reactions

               trajectory.append((self.time, self.state.copy()))

           return trajectory

Model Validation and Calibration
---------------------------------

Techniques for validating computational models against experimental data.

Parameter Estimation
~~~~~~~~~~~~~~~~~~~~

**Bayesian Parameter Inference:**

.. code-block:: python

   class BayesianParameterEstimation:
       def __init__(self, model, experimental_data):
           self.model = model
           self.data = experimental_data
           self.parameter_ranges = {}

       def set_parameter_prior(self, parameter_name, distribution):
           """Set prior distribution for parameter."""
           self.parameter_ranges[parameter_name] = distribution

       def log_likelihood(self, parameters):
           """Calculate log-likelihood of data given parameters."""
           # Run model with given parameters
           model_output = self.run_model(parameters)

           # Compare with experimental data
           log_likelihood = 0

           for observable in self.data.keys():
               model_values = model_output[observable]
               data_values = self.data[observable]
               data_errors = self.data.get(f"{observable}_error", 1.0)

               # Gaussian likelihood
               residuals = (model_values - data_values) / data_errors
               log_likelihood -= 0.5 * np.sum(residuals**2)

           return log_likelihood

       def log_prior(self, parameters):
           """Calculate log-prior probability of parameters."""
           log_prior = 0

           for param_name, value in parameters.items():
               if param_name in self.parameter_ranges:
                   distribution = self.parameter_ranges[param_name]
                   log_prior += distribution.logpdf(value)

           return log_prior

       def log_posterior(self, parameters):
           """Calculate log-posterior probability."""
           return self.log_likelihood(parameters) + self.log_prior(parameters)

       async def mcmc_sampling(self, n_samples=10000, burn_in=1000):
           """Markov Chain Monte Carlo sampling of posterior."""
           # Initialize chain
           current_params = self.initialize_parameters()
           current_log_posterior = self.log_posterior(current_params)

           samples = []
           log_posteriors = []

           for i in range(n_samples + burn_in):
               # Propose new parameters
               proposed_params = self.propose_parameters(current_params)
               proposed_log_posterior = self.log_posterior(proposed_params)

               # Metropolis-Hastings acceptance
               log_ratio = proposed_log_posterior - current_log_posterior

               if log_ratio > 0 or np.random.random() < np.exp(log_ratio):
                   # Accept proposal
                   current_params = proposed_params
                   current_log_posterior = proposed_log_posterior

               # Store sample (after burn-in)
               if i >= burn_in:
                   samples.append(current_params.copy())
                   log_posteriors.append(current_log_posterior)

           return samples, log_posteriors

Model Selection and Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Information Criteria:**

.. code-block:: python

   class ModelComparison:
       def __init__(self, models, data):
           self.models = models
           self.data = data

       def calculate_aic(self, model):
           """Calculate Akaike Information Criterion."""
           log_likelihood = model.log_likelihood(self.data)
           n_parameters = model.get_parameter_count()

           aic = 2 * n_parameters - 2 * log_likelihood
           return aic

       def calculate_bic(self, model):
           """Calculate Bayesian Information Criterion."""
           log_likelihood = model.log_likelihood(self.data)
           n_parameters = model.get_parameter_count()
           n_data_points = len(self.data)

           bic = n_parameters * np.log(n_data_points) - 2 * log_likelihood
           return bic

       def cross_validation_score(self, model, k_folds=5):
           """Calculate k-fold cross-validation score."""
           fold_size = len(self.data) // k_folds
           cv_scores = []

           for fold in range(k_folds):
               # Split data
               start_idx = fold * fold_size
               end_idx = (fold + 1) * fold_size

               test_data = self.data[start_idx:end_idx]
               train_data = np.concatenate([
                   self.data[:start_idx],
                   self.data[end_idx:]
               ])

               # Train model on training data
               model.fit(train_data)

               # Evaluate on test data
               score = model.evaluate(test_data)
               cv_scores.append(score)

           return np.mean(cv_scores), np.std(cv_scores)

       def model_evidence(self, model):
           """Calculate model evidence using thermodynamic integration."""
           # Simplified implementation
           # In practice, requires sophisticated numerical integration

           def log_likelihood_at_temperature(beta, parameters):
               return beta * model.log_likelihood(parameters, self.data)

           # Integrate over temperature parameter β from 0 to 1
           temperatures = np.linspace(0, 1, 100)
           evidence_estimates = []

           for beta in temperatures:
               # Sample parameters at this temperature
               samples = self.sample_at_temperature(model, beta)
               likelihood_values = [
                   log_likelihood_at_temperature(beta, params)
                   for params in samples
               ]
               evidence_estimates.append(np.mean(likelihood_values))

           # Numerical integration
           evidence = np.trapz(evidence_estimates, temperatures)
           return evidence

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

**Global Sensitivity Analysis:**

.. code-block:: python

   class SensitivityAnalysis:
       def __init__(self, model, parameters, output_metrics):
           self.model = model
           self.parameters = parameters
           self.output_metrics = output_metrics

       async def sobol_sensitivity(self, n_samples=1000):
           """Sobol sensitivity analysis."""
           from SALib.sample import saltelli
           from SALib.analyze import sobol

           # Define parameter problem
           problem = {
               'num_vars': len(self.parameters),
               'names': list(self.parameters.keys()),
               'bounds': [self.parameters[name]['bounds'] for name in self.parameters.keys()]
           }

           # Generate parameter samples
           param_values = saltelli.sample(problem, n_samples)

           # Evaluate model for all parameter combinations
           model_outputs = []

           for params in param_values:
               parameter_dict = {
                   name: params[i]
                   for i, name in enumerate(self.parameters.keys())
               }

               output = await self.model.run(parameter_dict)
               model_outputs.append([output[metric] for metric in self.output_metrics])

           model_outputs = np.array(model_outputs)

           # Calculate Sobol indices
           sensitivity_results = {}
           for i, metric in enumerate(self.output_metrics):
               Si = sobol.analyze(problem, model_outputs[:, i])
               sensitivity_results[metric] = Si

           return sensitivity_results

       def local_sensitivity(self, base_parameters, perturbation=0.01):
           """Local sensitivity analysis using finite differences."""
           base_output = self.model.run(base_parameters)

           sensitivities = {}

           for param_name in self.parameters:
               # Perturb parameter
               perturbed_params = base_parameters.copy()
               perturbed_params[param_name] *= (1 + perturbation)

               perturbed_output = self.model.run(perturbed_params)

               # Calculate sensitivities
               param_sensitivities = {}
               for metric in self.output_metrics:
                   sensitivity = (
                       (perturbed_output[metric] - base_output[metric]) /
                       (perturbed_params[param_name] - base_parameters[param_name]) *
                       base_parameters[param_name] / base_output[metric]
                   )
                   param_sensitivities[metric] = sensitivity

               sensitivities[param_name] = param_sensitivities

           return sensitivities

Performance Optimization
------------------------

Techniques for optimizing computational performance of morphogenetic models.

Parallelization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Spatial Domain Decomposition:**

.. code-block:: python

   import asyncio
   from concurrent.futures import ProcessPoolExecutor

   class ParallelMorphogeneticSimulation:
       def __init__(self, domain_size, n_processes=4):
           self.domain_size = domain_size
           self.n_processes = n_processes
           self.executor = ProcessPoolExecutor(max_workers=n_processes)

       def decompose_domain(self):
           """Decompose spatial domain into subdomains."""
           subdomains = []

           # Simple 2D decomposition
           rows = int(np.sqrt(self.n_processes))
           cols = self.n_processes // rows

           for i in range(rows):
               for j in range(cols):
                   subdomain = {
                       'x_start': i * self.domain_size[0] // rows,
                       'x_end': (i + 1) * self.domain_size[0] // rows,
                       'y_start': j * self.domain_size[1] // cols,
                       'y_end': (j + 1) * self.domain_size[1] // cols,
                       'process_id': i * cols + j
                   }
                   subdomains.append(subdomain)

           return subdomains

       async def parallel_step(self, agents, morphogen_fields):
           """Execute parallel simulation step."""
           subdomains = self.decompose_domain()

           # Partition agents by spatial location
           agent_partitions = self.partition_agents(agents, subdomains)
           field_partitions = self.partition_fields(morphogen_fields, subdomains)

           # Submit tasks to processes
           tasks = []
           for i, subdomain in enumerate(subdomains):
               task = self.executor.submit(
                   self.process_subdomain,
                   agent_partitions[i],
                   field_partitions[i],
                   subdomain
               )
               tasks.append(task)

           # Wait for completion
           results = await asyncio.gather(*[
               asyncio.wrap_future(task) for task in tasks
           ])

           # Merge results
           updated_agents, updated_fields = self.merge_results(results)

           # Handle boundary communications
           await self.synchronize_boundaries(updated_agents, updated_fields)

           return updated_agents, updated_fields

**GPU Acceleration:**

.. code-block:: python

   try:
       import cupy as cp
   except ImportError:
       cp = None

   class GPUAcceleratedModel:
       def __init__(self, use_gpu=True):
           self.use_gpu = use_gpu and cp is not None

           if self.use_gpu:
               self.xp = cp
           else:
               self.xp = np

       def gpu_reaction_diffusion_step(self, concentrations, dt):
           """GPU-accelerated reaction-diffusion step."""
           if not self.use_gpu:
               return self.cpu_reaction_diffusion_step(concentrations, dt)

           # Transfer data to GPU
           gpu_concentrations = {
               species: cp.asarray(conc)
               for species, conc in concentrations.items()
           }

           # GPU kernel for Laplacian calculation
           def laplacian_kernel(field):
               # Custom CUDA kernel for efficient Laplacian computation
               kernel = cp.RawKernel(r'''
               extern "C" __global__
               void laplacian_2d(const float* input, float* output,
                                int width, int height, float dx) {
                   int i = blockIdx.x * blockDim.x + threadIdx.x;
                   int j = blockIdx.y * blockDim.y + threadIdx.y;

                   if (i > 0 && i < width-1 && j > 0 && j < height-1) {
                       int idx = i * height + j;
                       output[idx] = (input[(i+1)*height + j] +
                                     input[(i-1)*height + j] +
                                     input[i*height + (j+1)] +
                                     input[i*height + (j-1)] -
                                     4.0f * input[idx]) / (dx * dx);
                   }
               }
               ''', 'laplacian_2d')

               output = cp.zeros_like(field)
               block_size = (16, 16)
               grid_size = ((field.shape[0] + block_size[0] - 1) // block_size[0],
                          (field.shape[1] + block_size[1] - 1) // block_size[1])

               kernel(grid_size, block_size, (field, output,
                     field.shape[0], field.shape[1], self.dx))

               return output

           # Update concentrations on GPU
           for species in gpu_concentrations:
               laplacian = laplacian_kernel(gpu_concentrations[species])
               reaction = self.gpu_reaction_terms[species](gpu_concentrations)

               gpu_concentrations[species] += dt * (
                   self.diffusion_coefficients[species] * laplacian + reaction
               )

           # Transfer results back to CPU
           cpu_concentrations = {
               species: cp.asnumpy(gpu_conc)
               for species, gpu_conc in gpu_concentrations.items()
           }

           return cpu_concentrations

Adaptive Time Stepping
~~~~~~~~~~~~~~~~~~~~~~

**Error-Based Step Size Control:**

.. code-block:: python

   class AdaptiveTimestepper:
       def __init__(self, initial_dt=0.001, tolerance=1e-6):
           self.dt = initial_dt
           self.tolerance = tolerance
           self.min_dt = 1e-8
           self.max_dt = 0.1

       async def adaptive_step(self, system_state, time_derivative_func):
           """Adaptive time step using embedded Runge-Kutta method."""
           while True:
               # Take full step
               k1 = await time_derivative_func(system_state)
               k2 = await time_derivative_func(
                   system_state + 0.5 * self.dt * k1
               )
               k3 = await time_derivative_func(
                   system_state + 0.5 * self.dt * k2
               )
               k4 = await time_derivative_func(
                   system_state + self.dt * k3
               )

               full_step = system_state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

               # Take two half steps
               half_k1 = k1
               half_k2 = await time_derivative_func(
                   system_state + 0.25 * self.dt * half_k1
               )
               half_k3 = await time_derivative_func(
                   system_state + 0.25 * self.dt * half_k2
               )
               half_k4 = await time_derivative_func(
                   system_state + 0.5 * self.dt * half_k3
               )

               half_step1 = system_state + (self.dt / 12) * (half_k1 + 2*half_k2 + 2*half_k3 + half_k4)

               # Second half step
               half2_k1 = await time_derivative_func(half_step1)
               half2_k2 = await time_derivative_func(
                   half_step1 + 0.25 * self.dt * half2_k1
               )
               half2_k3 = await time_derivative_func(
                   half_step1 + 0.25 * self.dt * half2_k2
               )
               half2_k4 = await time_derivative_func(
                   half_step1 + 0.5 * self.dt * half2_k3
               )

               half_step2 = half_step1 + (self.dt / 12) * (half2_k1 + 2*half2_k2 + 2*half2_k3 + half2_k4)

               # Estimate error
               error = np.linalg.norm(half_step2 - full_step)

               if error < self.tolerance:
                   # Accept step
                   new_dt = min(self.max_dt, 0.9 * self.dt * (self.tolerance / error) ** 0.25)
                   self.dt = max(self.min_dt, new_dt)
                   return half_step2

               else:
                   # Reject step, reduce time step
                   self.dt = max(self.min_dt, 0.5 * self.dt)

                   if self.dt <= self.min_dt:
                       # Force acceptance at minimum time step
                       return full_step

Conclusion
----------

Computational modeling in morphogenesis requires careful consideration of:

**Model Architecture:**
- Choosing appropriate modeling paradigms (agent-based, continuum, hybrid)
- Balancing biological realism with computational efficiency
- Incorporating multiple scales and stochastic effects

**Implementation Considerations:**
- Efficient algorithms and data structures
- Parallelization and GPU acceleration
- Adaptive spatial and temporal resolution
- Memory management for large-scale simulations

**Validation and Analysis:**
- Parameter estimation from experimental data
- Model comparison and selection
- Sensitivity analysis and uncertainty quantification
- Performance optimization and scalability

**Best Practices:**
- Modular design for flexibility and reusability
- Comprehensive testing and validation
- Documentation of assumptions and limitations
- Version control and reproducibility measures

The field continues to evolve with advances in computational power, algorithmic development, and experimental techniques, enabling increasingly sophisticated and predictive models of morphogenetic processes.