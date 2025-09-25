Advanced Examples
=================

This section provides sophisticated examples demonstrating advanced features of the morphogenesis platform, including multi-scale modeling, complex behaviors, and research-level applications.

Multi-Scale Morphogenetic System
---------------------------------

This example demonstrates a multi-scale model that integrates molecular, cellular, and tissue-level processes.

**Overview:**
Simulate embryonic neural tube closure, incorporating gene regulatory networks, cellular behaviors, and tissue-level mechanical forces.

**Implementation:**

.. code-block:: python

   import asyncio
   import numpy as np
   from core.coordination.coordinator import DeterministicCoordinator
   from core.agents.behaviors.morphogen_cell import MorphogenCell
   from analysis.visualization.animated_morphogenesis_visualizer import AnimatedVisualizer

   class NeuralTubeClosureModel:
       def __init__(self, tissue_dimensions=(100, 200)):
           self.tissue_dimensions = tissue_dimensions
           self.coordinator = DeterministicCoordinator(
               grid_size=tissue_dimensions,
               max_agents=5000
           )

           # Multi-scale components
           self.gene_networks = {}
           self.morphogen_fields = {}
           self.mechanical_forces = {}

       async def initialize_neural_plate(self):
           """Initialize neural plate with specified cell types."""
           cell_types = {
               'neural_progenitor': {'count': 2000, 'color': 'blue'},
               'neural_crest': {'count': 800, 'color': 'green'},
               'surface_ectoderm': {'count': 1200, 'color': 'yellow'}
           }

           agents = []
           agent_id = 0

           for cell_type, properties in cell_types.items():
               for _ in range(properties['count']):
                   # Spatially organize different cell types
                   position = self.get_cell_type_position(cell_type, agent_id)

                   # Create specialized agent
                   agent = NeuralTubeCell(
                       agent_id=agent_id,
                       initial_position=position,
                       cell_type=cell_type
                   )

                   # Initialize gene regulatory network
                   await self.setup_gene_network(agent, cell_type)

                   agents.append(agent)
                   agent_id += 1

               await self.coordinator.add_agents(agents)

           return agents

       async def setup_morphogen_fields(self):
           """Initialize morphogen fields that guide neural tube closure."""
           morphogens = ['bmp4', 'shh', 'wnt', 'fgf8']

           for morphogen in morphogens:
               field = MorphogenField(
                   name=morphogen,
                   domain_size=self.tissue_dimensions,
                   diffusion_coefficient=self.get_diffusion_coefficient(morphogen),
                   degradation_rate=self.get_degradation_rate(morphogen)
               )

               # Set initial conditions based on experimental data
               await self.set_morphogen_initial_conditions(field, morphogen)

               self.morphogen_fields[morphogen] = field

       async def simulate_neural_tube_closure(self, simulation_time=48.0):
           """Complete neural tube closure simulation."""
           print("Starting neural tube closure simulation...")

           # Initialize system
           agents = await self.initialize_neural_plate()
           await self.setup_morphogen_fields()

           # Setup visualization
           visualizer = AnimatedVisualizer(
               save_path="neural_tube_closure.mp4",
               fps=10
           )

           # Main simulation loop
           timestep = 0.1  # hours
           current_time = 0.0

           simulation_data = []

           while current_time < simulation_time:
               # Update gene regulatory networks
               await self.update_gene_networks(agents, current_time)

               # Update morphogen fields
               await self.update_morphogen_fields(timestep)

               # Calculate mechanical forces
               await self.calculate_tissue_mechanics(agents)

               # Agent decision-making and actions
               await self.coordinator.step()

               # Monitor closure progress
               closure_metrics = await self.assess_closure_progress(agents)

               # Record data
               if timestep % 1.0 < 0.1:  # Record every hour
                   frame_data = {
                       'time': current_time,
                       'agent_positions': [(a.position.x, a.position.y) for a in agents],
                       'agent_types': [a.cell_type for a in agents],
                       'closure_progress': closure_metrics['closure_percentage'],
                       'neural_fold_height': closure_metrics['fold_height']
                   }
                   simulation_data.append(frame_data)

                   # Add frame to visualization
                   await visualizer.add_frame(frame_data)

               current_time += timestep

               # Check for completion
               if closure_metrics['closure_percentage'] > 95.0:
                   print(f"Neural tube closure completed at t={current_time:.1f}h")
                   break

           # Generate final visualization
           await visualizer.finalize()

           return simulation_data

   class NeuralTubeCell(MorphogenCell):
       def __init__(self, agent_id, initial_position, cell_type):
           super().__init__(agent_id, initial_position, cell_type)

           self.gene_expression = {}
           self.mechanical_properties = self.initialize_mechanical_properties()
           self.apical_constriction = 0.0

       async def gene_network_step(self, dt, morphogen_concentrations):
           """Update gene regulatory network."""
           # Simplified gene network for neural tube closure
           genes = ['pax3', 'msx1', 'bmp4', 'shh']

           for gene in genes:
               current_expression = self.gene_expression.get(gene, 0.0)

               # Calculate regulatory inputs
               regulatory_input = 0.0

               if gene == 'pax3':
                   # PAX3 activated by WNT, repressed by BMP4
                   wnt_signal = morphogen_concentrations.get('wnt', 0.0)
                   bmp4_signal = morphogen_concentrations.get('bmp4', 0.0)
                   regulatory_input = 2.0 * wnt_signal - 1.5 * bmp4_signal

               elif gene == 'msx1':
                   # MSX1 activated by BMP4
                   bmp4_signal = morphogen_concentrations.get('bmp4', 0.0)
                   regulatory_input = 1.8 * bmp4_signal

               # Update expression level
               new_expression = current_expression + dt * (
                   regulatory_input - 0.1 * current_expression  # Decay
               )

               self.gene_expression[gene] = max(0.0, new_expression)

       async def mechanical_behavior_step(self):
           """Update mechanical behavior based on gene expression."""
           # Apical constriction based on gene expression
           pax3_level = self.gene_expression.get('pax3', 0.0)
           if pax3_level > 0.5 and self.cell_type == 'neural_progenitor':
               # Increase apical constriction
               self.apical_constriction = min(1.0, self.apical_constriction + 0.01)

               # Apply constriction force
               constriction_force = self.calculate_apical_constriction_force()
               await self.apply_force(constriction_force)

       async def cell_fate_decision(self, morphogen_concentrations):
           """Make cell fate decisions based on morphogen gradients."""
           shh_level = morphogen_concentrations.get('shh', 0.0)
           bmp4_level = morphogen_concentrations.get('bmp4', 0.0)

           if self.cell_type == 'neural_progenitor':
               if shh_level > 0.7:
                   # High SHH -> ventral neural fate
                   await self.differentiate_to('ventral_neuron')
               elif bmp4_level > 0.6:
                   # High BMP4 -> neural crest fate
                   await self.differentiate_to('neural_crest')

   # Example usage
   async def run_neural_tube_example():
       model = NeuralTubeClosureModel(tissue_dimensions=(150, 300))

       # Run simulation
       results = await model.simulate_neural_tube_closure(simulation_time=36.0)

       # Analyze results
       closure_times = [data['time'] for data in results
                       if data['closure_progress'] > 90.0]

       if closure_times:
           print(f"Neural tube closure achieved at t={min(closure_times):.1f} hours")
       else:
           print("Neural tube closure not completed within simulation time")

       return results

   # Run the example
   if __name__ == "__main__":
       results = asyncio.run(run_neural_tube_example())

Collective Intelligence Emergence
----------------------------------

This example demonstrates emergent collective intelligence in cellular populations solving complex spatial problems.

**Overview:**
Simulate a population of cells that must collectively solve a maze navigation task through local communication and learning.

**Implementation:**

.. code-block:: python

   from core.agents.behaviors.adaptive_cell import AdaptiveCell
   from core.coordination.spatial_index import SpatialIndex
   import random

   class CollectiveIntelligenceExperiment:
       def __init__(self, maze_size=(50, 50), population_size=200):
           self.maze_size = maze_size
           self.population_size = population_size
           self.maze = self.generate_maze()
           self.coordinator = DeterministicCoordinator(
               grid_size=maze_size,
               max_agents=population_size
           )

       def generate_maze(self):
           """Generate a challenging maze with multiple paths."""
           maze = np.zeros(self.maze_size)

           # Add walls (1 = wall, 0 = open space)
           # Create complex maze structure
           for i in range(self.maze_size[0]):
               for j in range(self.maze_size[1]):
                   if (i % 4 == 0 and j % 4 != 2) or (j % 4 == 0 and i % 4 != 2):
                       maze[i, j] = 1

           # Ensure start and goal are accessible
           maze[0, 0] = 0  # Start position
           maze[-1, -1] = 0  # Goal position

           return maze

       async def initialize_intelligent_swarm(self):
           """Initialize swarm of collectively intelligent cells."""
           agents = []

           for agent_id in range(self.population_size):
               # Start all agents near the maze entrance
               start_positions = [(1, 1), (2, 1), (1, 2), (2, 2)]
               position = random.choice(start_positions)

               agent = CollectiveIntelligenceCell(
                   agent_id=agent_id,
                   initial_position=position,
                   maze=self.maze,
                   goal_position=(self.maze_size[0]-1, self.maze_size[1]-1)
               )

               agents.append(agent)

           await self.coordinator.add_agents(agents)
           return agents

       async def simulate_collective_problem_solving(self, max_time=1000):
           """Simulate collective maze solving."""
           agents = await self.initialize_intelligent_swarm()

           # Metrics to track
           solution_discovery_time = None
           collective_learning_curve = []
           communication_network_evolution = []

           for timestep in range(max_time):
               # Update all agents
               await self.coordinator.step()

               # Analyze collective intelligence metrics
               metrics = await self.analyze_collective_intelligence(agents, timestep)
               collective_learning_curve.append(metrics)

               # Check if solution has been discovered
               if not solution_discovery_time:
                   goal_reached_agents = [a for a in agents if a.reached_goal]
                   if goal_reached_agents:
                       solution_discovery_time = timestep
                       print(f"Solution discovered at timestep {timestep}")

               # Measure communication network
               network_metrics = await self.analyze_communication_network(agents)
               communication_network_evolution.append(network_metrics)

               # Check convergence
               if metrics['collective_efficiency'] > 0.9:
                   print(f"Collective intelligence converged at timestep {timestep}")
                   break

           return {
               'solution_discovery_time': solution_discovery_time,
               'learning_curve': collective_learning_curve,
               'communication_evolution': communication_network_evolution,
               'final_paths': [agent.path_history for agent in agents]
           }

       async def analyze_collective_intelligence(self, agents, timestep):
           """Analyze emergence of collective intelligence."""
           # Path efficiency
           path_efficiencies = []
           for agent in agents:
               if len(agent.path_history) > 1:
                   efficiency = agent.calculate_path_efficiency()
                   path_efficiencies.append(efficiency)

           # Information sharing effectiveness
           shared_knowledge_coverage = await self.calculate_knowledge_coverage(agents)

           # Coordination level
           coordination_score = await self.calculate_coordination_score(agents)

           # Collective intelligence metrics
           metrics = {
               'timestep': timestep,
               'mean_path_efficiency': np.mean(path_efficiencies) if path_efficiencies else 0,
               'knowledge_coverage': shared_knowledge_coverage,
               'coordination_score': coordination_score,
               'collective_efficiency': (
                   np.mean(path_efficiencies) * shared_knowledge_coverage * coordination_score
               ) if path_efficiencies else 0
           }

           return metrics

   class CollectiveIntelligenceCell(AdaptiveCell):
       def __init__(self, agent_id, initial_position, maze, goal_position):
           super().__init__(agent_id, initial_position)
           self.maze = maze
           self.goal_position = goal_position
           self.knowledge_map = {}  # Learned information about maze
           self.pheromone_trails = {}  # Chemical communication
           self.path_history = [initial_position]
           self.reached_goal = False

       async def explore_and_learn(self):
           """Explore maze and learn about structure."""
           # Sense local environment
           local_observations = await self.sense_local_maze_structure()

           # Update knowledge map
           for observation in local_observations:
               position, maze_value = observation
               self.knowledge_map[position] = maze_value

           # Share knowledge with nearby agents
           neighbors = await self.get_neighbors(radius=3)
           for neighbor in neighbors:
               await self.share_knowledge(neighbor)

       async def collective_decision_making(self):
           """Make movement decisions based on collective knowledge."""
           possible_moves = await self.get_valid_moves()

           if not possible_moves:
               return None

           # Evaluate moves based on multiple criteria
           move_scores = {}

           for move in possible_moves:
               new_position = (self.position[0] + move[0], self.position[1] + move[1])

               # Individual knowledge
               individual_score = self.evaluate_move_individual_knowledge(new_position)

               # Collective knowledge
               collective_score = await self.evaluate_move_collective_knowledge(new_position)

               # Pheromone trails
               pheromone_score = self.evaluate_pheromone_trails(new_position)

               # Exploration vs exploitation
               exploration_score = self.evaluate_exploration_value(new_position)

               # Combined score
               total_score = (
                   0.3 * individual_score +
                   0.4 * collective_score +
                   0.2 * pheromone_score +
                   0.1 * exploration_score
               )

               move_scores[move] = total_score

           # Select best move
           best_move = max(move_scores.keys(), key=lambda m: move_scores[m])
           return best_move

       async def deposit_pheromones(self):
           """Deposit chemical trails for communication."""
           # Deposit different types of pheromones
           if self.reached_goal:
               # Success pheromone - very strong
               pheromone_strength = 10.0
               pheromone_type = 'success'
           else:
               # Exploration pheromone - weaker
               distance_to_goal = self.calculate_distance_to_goal()
               pheromone_strength = max(0.1, 2.0 / (1.0 + distance_to_goal))
               pheromone_type = 'exploration'

           await self.coordinator.deposit_pheromone(
               position=self.position,
               pheromone_type=pheromone_type,
               strength=pheromone_strength
           )

       async def adaptive_learning_step(self):
           """Adapt behavior based on experience."""
           # Update path efficiency estimator
           if len(self.path_history) > 10:
               recent_efficiency = self.calculate_recent_path_efficiency()

               if recent_efficiency < self.previous_efficiency:
                   # Decrease exploration, increase exploitation
                   self.exploration_rate *= 0.95
               else:
                   # Increase exploration
                   self.exploration_rate *= 1.05

               self.exploration_rate = np.clip(self.exploration_rate, 0.01, 0.5)

   # Example usage
   async def run_collective_intelligence_example():
       experiment = CollectiveIntelligenceExperiment(
           maze_size=(30, 30),
           population_size=100
       )

       results = await experiment.simulate_collective_problem_solving(max_time=2000)

       print(f"Solution discovered at timestep: {results['solution_discovery_time']}")
       print(f"Final collective efficiency: {results['learning_curve'][-1]['collective_efficiency']:.3f}")

       return results

   # Run the example
   if __name__ == "__main__":
       results = asyncio.run(run_collective_intelligence_example())

Morphogen-Driven Pattern Formation
-----------------------------------

This example demonstrates complex pattern formation using multiple interacting morphogen systems.

**Overview:**
Simulate Drosophila wing disc development with multiple morphogen gradients creating precise spatial patterns.

**Implementation:**

.. code-block:: python

   from core.data.types import Position2D
   from analysis.statistical.descriptive import DescriptiveAnalyzer

   class WingDiscDevelopmentModel:
       def __init__(self, disc_size=(100, 100)):
           self.disc_size = disc_size
           self.morphogens = {}
           self.cells = []
           self.pattern_regions = {}

       async def initialize_wing_disc(self):
           """Initialize wing imaginal disc."""
           # Create morphogen fields
           morphogen_names = ['hedgehog', 'wingless', 'decapentaplegic', 'engrailed']

           for name in morphogen_names:
               morphogen = MorphogenField(
                   name=name,
                   domain_size=self.disc_size,
                   diffusion_coefficient=self.get_morphogen_properties(name)['diffusion'],
                   degradation_rate=self.get_morphogen_properties(name)['degradation']
               )
               self.morphogens[name] = morphogen

           # Initialize cell population
           cell_density = 0.8
           n_cells = int(self.disc_size[0] * self.disc_size[1] * cell_density)

           for cell_id in range(n_cells):
               position = self.generate_random_position_in_disc()
               cell = WingDiscCell(cell_id, position)
               self.cells.append(cell)

           # Set up initial morphogen sources
           await self.establish_initial_morphogen_sources()

       async def establish_initial_morphogen_sources(self):
           """Set up initial morphogen expression patterns."""
           center_x, center_y = self.disc_size[0] // 2, self.disc_size[1] // 2

           # Hedgehog: expressed in posterior compartment
           for i in range(center_x, self.disc_size[0]):
               for j in range(self.disc_size[1]):
                   if self.is_in_disc(i, j):
                       self.morphogens['hedgehog'].set_production_rate((i, j), 1.0)

           # Wingless: expressed along anterior-posterior boundary
           for j in range(self.disc_size[1]):
               if self.is_in_disc(center_x, j):
                   self.morphogens['wingless'].set_production_rate((center_x, j), 0.8)

           # Decapentaplegic: expressed along anterior-posterior boundary
           for i in range(center_x-2, center_x+3):
               for j in range(self.disc_size[1]):
                   if self.is_in_disc(i, j):
                       self.morphogens['decapentaplegic'].set_production_rate((i, j), 0.6)

       async def simulate_wing_development(self, development_time=100):
           """Simulate wing disc development and pattern formation."""
           print("Starting wing disc development simulation...")

           pattern_evolution = []

           for timestep in range(development_time):
               # Update morphogen fields
               for morphogen in self.morphogens.values():
                   await morphogen.step(dt=0.1)

               # Cell fate specification
               await self.update_cell_fates()

               # Cell division and growth
               await self.cell_division_step()

               # Record pattern every 10 timesteps
               if timestep % 10 == 0:
                   pattern_data = await self.analyze_current_pattern()
                   pattern_evolution.append(pattern_data)

                   print(f"Timestep {timestep}: {len(self.pattern_regions)} pattern regions identified")

           return pattern_evolution

       async def update_cell_fates(self):
           """Update cell fates based on morphogen concentrations."""
           for cell in self.cells:
               # Sample morphogen concentrations at cell position
               local_concentrations = {}
               for name, morphogen in self.morphogens.items():
                   concentration = morphogen.get_concentration_at(cell.position)
                   local_concentrations[name] = concentration

               # Cell fate decision based on morphogen thresholds
               await cell.determine_cell_fate(local_concentrations)

       async def analyze_current_pattern(self):
           """Analyze current spatial pattern of cell fates."""
           # Group cells by fate
           fate_groups = {}
           for cell in self.cells:
               fate = cell.current_fate
               if fate not in fate_groups:
                   fate_groups[fate] = []
               fate_groups[fate].append(cell)

           # Analyze spatial organization
           pattern_metrics = {}

           for fate, cell_group in fate_groups.items():
               positions = [cell.position for cell in cell_group]

               # Calculate clustering
               clustering_score = self.calculate_spatial_clustering(positions)

               # Calculate boundary sharpness
               boundary_sharpness = self.calculate_boundary_sharpness(fate, positions)

               pattern_metrics[fate] = {
                   'cell_count': len(cell_group),
                   'clustering_score': clustering_score,
                   'boundary_sharpness': boundary_sharpness
               }

           return pattern_metrics

   class WingDiscCell:
       def __init__(self, cell_id, initial_position):
           self.cell_id = cell_id
           self.position = initial_position
           self.current_fate = 'undetermined'
           self.gene_expression = {}
           self.division_timer = 0

       async def determine_cell_fate(self, morphogen_concentrations):
           """Determine cell fate based on morphogen levels."""
           hh = morphogen_concentrations.get('hedgehog', 0)
           wg = morphogen_concentrations.get('wingless', 0)
           dpp = morphogen_concentrations.get('decapentaplegic', 0)

           # Simplified fate determination logic
           if hh > 0.7 and wg > 0.5:
               self.current_fate = 'wing_margin'
           elif dpp > 0.6:
               self.current_fate = 'wing_pouch'
           elif hh > 0.5:
               self.current_fate = 'posterior_compartment'
           elif wg > 0.3:
               self.current_fate = 'wing_hinge'
           else:
               self.current_fate = 'anterior_compartment'

           # Update gene expression based on fate
           await self.update_gene_expression()

       async def update_gene_expression(self):
           """Update gene expression patterns based on cell fate."""
           fate_gene_programs = {
               'wing_margin': {'cut': 1.0, 'notch': 0.8},
               'wing_pouch': {'spalt': 1.0, 'optomotor_blind': 0.6},
               'posterior_compartment': {'engrailed': 1.0, 'hedgehog': 0.9},
               'wing_hinge': {'teashirt': 1.0, 'homothorax': 0.7},
               'anterior_compartment': {'cubitus_interruptus': 0.5}
           }

           if self.current_fate in fate_gene_programs:
               self.gene_expression.update(fate_gene_programs[self.current_fate])

   # Example usage
   async def run_wing_pattern_example():
       model = WingDiscDevelopmentModel(disc_size=(80, 80))
       await model.initialize_wing_disc()

       pattern_evolution = await model.simulate_wing_development(development_time=150)

       # Analyze final pattern
       final_pattern = pattern_evolution[-1]
       print("Final pattern regions:")
       for fate, metrics in final_pattern.items():
           print(f"  {fate}: {metrics['cell_count']} cells, "
                 f"clustering={metrics['clustering_score']:.3f}")

       return pattern_evolution

   # Run the example
   if __name__ == "__main__":
       results = asyncio.run(run_wing_pattern_example())

Hybrid Continuous-Discrete Modeling
------------------------------------

This example demonstrates integration of continuous field equations with discrete cellular agents.

**Overview:**
Model tumor growth with continuous nutrient fields, discrete cancer cells, and immune system response.

**Implementation:**

.. code-block:: python

   from scipy.ndimage import gaussian_filter
   import matplotlib.pyplot as plt

   class TumorGrowthModel:
       def __init__(self, domain_size=(200, 200)):
           self.domain_size = domain_size
           self.dx = 0.1  # Spatial resolution (mm)
           self.dt = 0.1  # Time step (hours)

           # Continuous fields
           self.nutrient_field = np.ones(domain_size)
           self.oxygen_field = np.ones(domain_size)
           self.drug_field = np.zeros(domain_size)

           # Discrete agents
           self.cancer_cells = []
           self.immune_cells = []

           # Model parameters
           self.nutrient_diffusion = 0.1
           self.oxygen_diffusion = 0.2
           self.drug_diffusion = 0.05

       async def initialize_tumor(self, initial_tumor_size=10):
           """Initialize tumor with cancer cells and immune response."""
           center = (self.domain_size[0] // 2, self.domain_size[1] // 2)

           # Initialize cancer cells in a small cluster
           n_cancer_cells = initial_tumor_size**2
           for cell_id in range(n_cancer_cells):
               # Random position within initial tumor region
               offset_x = np.random.randint(-initial_tumor_size//2, initial_tumor_size//2)
               offset_y = np.random.randint(-initial_tumor_size//2, initial_tumor_size//2)

               position = (center[0] + offset_x, center[1] + offset_y)
               position = self.clamp_position(position)

               cancer_cell = CancerCell(
                   agent_id=cell_id,
                   initial_position=position,
                   tumor_model=self
               )

               self.cancer_cells.append(cancer_cell)

           # Initialize immune cells throughout domain
           n_immune_cells = 50
           for cell_id in range(n_immune_cells):
               position = (
                   np.random.randint(0, self.domain_size[0]),
                   np.random.randint(0, self.domain_size[1])
               )

               immune_cell = ImmuneCell(
                   agent_id=cell_id + n_cancer_cells,
                   initial_position=position,
                   tumor_model=self
               )

               self.immune_cells.append(immune_cell)

       async def update_continuous_fields(self):
           """Update continuous nutrient, oxygen, and drug fields."""
           # Nutrient field dynamics
           nutrient_laplacian = self.calculate_laplacian(self.nutrient_field)
           nutrient_consumption = self.calculate_nutrient_consumption()

           self.nutrient_field += self.dt * (
               self.nutrient_diffusion * nutrient_laplacian
               - nutrient_consumption
               + self.nutrient_supply_rate
           )

           # Oxygen field dynamics
           oxygen_laplacian = self.calculate_laplacian(self.oxygen_field)
           oxygen_consumption = self.calculate_oxygen_consumption()

           self.oxygen_field += self.dt * (
               self.oxygen_diffusion * oxygen_laplacian
               - oxygen_consumption
               + self.oxygen_supply_rate
           )

           # Drug field dynamics (if treatment active)
           if self.drug_treatment_active:
               drug_laplacian = self.calculate_laplacian(self.drug_field)
               drug_uptake = self.calculate_drug_uptake()

               self.drug_field += self.dt * (
                   self.drug_diffusion * drug_laplacian
                   - drug_uptake
                   + self.drug_injection_rate
               )

           # Ensure non-negative concentrations
           self.nutrient_field = np.maximum(0, self.nutrient_field)
           self.oxygen_field = np.maximum(0, self.oxygen_field)
           self.drug_field = np.maximum(0, self.drug_field)

       def calculate_nutrient_consumption(self):
           """Calculate nutrient consumption by cells."""
           consumption = np.zeros(self.domain_size)

           for cell in self.cancer_cells:
               i, j = cell.position
               if 0 <= i < self.domain_size[0] and 0 <= j < self.domain_size[1]:
                   consumption[i, j] += cell.nutrient_consumption_rate

           return consumption

       async def simulate_tumor_growth(self, simulation_time=100, treatment_start_time=50):
           """Simulate tumor growth with optional treatment."""
           print("Starting tumor growth simulation...")

           # Initialize system
           await self.initialize_tumor()

           # Tracking variables
           tumor_size_history = []
           immune_response_history = []
           drug_effectiveness_history = []

           # Main simulation loop
           for timestep in range(int(simulation_time / self.dt)):
               current_time = timestep * self.dt

               # Update continuous fields
               await self.update_continuous_fields()

               # Start drug treatment at specified time
               if current_time >= treatment_start_time:
                   self.drug_treatment_active = True
                   self.drug_injection_rate = 0.1

               # Update cancer cells
               cancer_cell_updates = []
               for cell in self.cancer_cells:
                   update_task = cell.step(self.dt)
                   cancer_cell_updates.append(update_task)

               await asyncio.gather(*cancer_cell_updates)

               # Update immune cells
               immune_cell_updates = []
               for cell in self.immune_cells:
                   update_task = cell.step(self.dt)
                   immune_cell_updates.append(update_task)

               await asyncio.gather(*immune_cell_updates)

               # Handle cell division and death
               await self.process_cell_division_and_death()

               # Record metrics every hour
               if timestep % int(1.0 / self.dt) == 0:
                   tumor_size = len(self.cancer_cells)
                   immune_response = self.calculate_immune_response_strength()
                   drug_effectiveness = self.calculate_drug_effectiveness()

                   tumor_size_history.append(tumor_size)
                   immune_response_history.append(immune_response)
                   drug_effectiveness_history.append(drug_effectiveness)

                   print(f"t={current_time:.1f}h: Tumor size={tumor_size}, "
                         f"Immune response={immune_response:.3f}")

           return {
               'tumor_size': tumor_size_history,
               'immune_response': immune_response_history,
               'drug_effectiveness': drug_effectiveness_history,
               'final_nutrient_field': self.nutrient_field.copy(),
               'final_drug_field': self.drug_field.copy()
           }

   class CancerCell:
       def __init__(self, agent_id, initial_position, tumor_model):
           self.agent_id = agent_id
           self.position = initial_position
           self.tumor_model = tumor_model
           self.energy = 1.0
           self.division_threshold = 1.5
           self.death_threshold = 0.1
           self.drug_resistance = np.random.uniform(0.1, 0.9)

       async def step(self, dt):
           """Update cancer cell state."""
           # Sample local environment
           local_nutrient = self.sample_field(self.tumor_model.nutrient_field)
           local_oxygen = self.sample_field(self.tumor_model.oxygen_field)
           local_drug = self.sample_field(self.tumor_model.drug_field)

           # Metabolism and growth
           growth_rate = min(local_nutrient, local_oxygen) * 0.1
           drug_damage = local_drug * (1.0 - self.drug_resistance) * 0.2

           self.energy += dt * (growth_rate - drug_damage - 0.05)  # Baseline maintenance cost

           # Movement (simplified random walk with chemotaxis)
           nutrient_gradient = self.calculate_nutrient_gradient()
           movement = self.calculate_movement_direction(nutrient_gradient)
           await self.move(movement)

       def sample_field(self, field):
           """Sample continuous field at cell position."""
           i, j = self.position
           if 0 <= i < field.shape[0] and 0 <= j < field.shape[1]:
               return field[i, j]
           return 0.0

   class ImmuneCell:
       def __init__(self, agent_id, initial_position, tumor_model):
           self.agent_id = agent_id
           self.position = initial_position
           self.tumor_model = tumor_model
           self.activation_level = 0.0
           self.target_cancer_cell = None

       async def step(self, dt):
           """Update immune cell state."""
           # Detect nearby cancer cells
           nearby_cancer_cells = self.detect_cancer_cells(radius=5)

           if nearby_cancer_cells:
               # Become activated
               self.activation_level = min(1.0, self.activation_level + 0.1 * dt)

               # Target closest cancer cell
               self.target_cancer_cell = min(
                   nearby_cancer_cells,
                   key=lambda cell: self.distance_to(cell)
               )

               # Move toward target
               direction = self.calculate_direction_to_target()
               await self.move(direction)

               # Attack if close enough
               if self.distance_to(self.target_cancer_cell) < 2:
                   await self.attack_cancer_cell(self.target_cancer_cell)

           else:
               # Random patrolling
               self.activation_level = max(0.0, self.activation_level - 0.05 * dt)
               random_direction = self.generate_random_direction()
               await self.move(random_direction)

   # Example usage
   async def run_tumor_growth_example():
       model = TumorGrowthModel(domain_size=(150, 150))

       results = await model.simulate_tumor_growth(
           simulation_time=200,  # hours
           treatment_start_time=100  # hours
       )

       # Plot results
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))

       # Tumor growth curve
       time_points = np.arange(len(results['tumor_size']))
       axes[0, 0].plot(time_points, results['tumor_size'])
       axes[0, 0].set_xlabel('Time (hours)')
       axes[0, 0].set_ylabel('Tumor Size (cell count)')
       axes[0, 0].set_title('Tumor Growth')

       # Immune response
       axes[0, 1].plot(time_points, results['immune_response'])
       axes[0, 1].set_xlabel('Time (hours)')
       axes[0, 1].set_ylabel('Immune Response Strength')
       axes[0, 1].set_title('Immune System Activity')

       # Final nutrient field
       im1 = axes[1, 0].imshow(results['final_nutrient_field'], cmap='viridis')
       axes[1, 0].set_title('Final Nutrient Distribution')
       plt.colorbar(im1, ax=axes[1, 0])

       # Final drug field
       im2 = axes[1, 1].imshow(results['final_drug_field'], cmap='plasma')
       axes[1, 1].set_title('Final Drug Concentration')
       plt.colorbar(im2, ax=axes[1, 1])

       plt.tight_layout()
       plt.savefig('tumor_growth_results.png', dpi=300)
       plt.show()

       return results

   # Run the example
   if __name__ == "__main__":
       results = asyncio.run(run_tumor_growth_example())

Conclusion
----------

These advanced examples demonstrate the power and flexibility of the morphogenesis platform for studying complex biological systems. Key features highlighted include:

**Multi-Scale Integration:**
- Gene regulatory networks controlling cellular behavior
- Tissue-level mechanical forces influencing individual cells
- Seamless coupling between different biological scales

**Emergent Intelligence:**
- Collective problem-solving without centralized control
- Learning and adaptation at both individual and group levels
- Communication networks facilitating information sharing

**Pattern Formation:**
- Multiple morphogen gradients creating spatial patterns
- Cell fate specification based on positional information
- Robust pattern formation despite noise and perturbations

**Hybrid Modeling:**
- Continuous fields for diffusible factors
- Discrete agents for individual cellular behavior
- Efficient computational methods for large-scale simulations

These examples provide a foundation for researchers to develop their own sophisticated morphogenesis models, adapt the platform to specific biological questions, and explore the frontiers of developmental biology through computational simulation.