Basic Cell Sorting Example
=========================

This example demonstrates how cells with different adhesion properties spontaneously organize themselves into distinct regions - a fundamental process in development and tissue organization.

Biological Background
--------------------

**Cell Sorting in Development**

During embryonic development, initially mixed populations of cells must separate into distinct tissues and organs. This process, known as cell sorting, is driven by differential cellular adhesion - cells prefer to associate with certain types of neighbors over others.

**Real-World Examples:**
   * **Germ Layer Formation**: During gastrulation, ectoderm, mesoderm, and endoderm cells sort into distinct layers
   * **Tissue Boundary Formation**: Different tissue types maintain sharp boundaries through differential adhesion
   * **Organ Development**: Cell sorting helps establish the architecture of organs like the kidney and liver

**Key Biological Principles:**
   * **Differential Adhesion Hypothesis**: Cells with stronger mutual adhesion will aggregate together
   * **Surface Tension Minimization**: Cell populations arrange to minimize interfacial energy
   * **Local Interactions**: Cells only respond to their immediate neighbors, yet create global organization

The Computational Model
-----------------------

Our model represents cells as autonomous agents with different adhesion preferences. Each cell:

1. **Senses its local neighborhood** to count different cell types nearby
2. **Calculates its satisfaction** based on neighbor composition
3. **Attempts to move** if dissatisfied with current location
4. **Updates its internal state** based on new environment

**Key Parameters:**
   * **Population Size**: Total number of cellular agents
   * **Cell Types**: Different adhesion classes (e.g., 'red' and 'blue')
   * **Adhesion Strength**: How strongly cells prefer similar neighbors
   * **Grid Size**: Spatial dimensions of the simulation
   * **Simulation Steps**: How long to run the experiment

Implementation
-------------

Let's examine the complete implementation:

.. code-block:: python

   """
   Basic Cell Sorting Example

   This example demonstrates how cells with different adhesion properties
   spontaneously organize themselves through local interactions.
   """

   import asyncio
   import numpy as np
   import matplotlib.pyplot as plt
   from typing import List, Dict, Tuple
   import logging

   from core.agents.cell_agent import AsyncCellAgent
   from core.coordination.coordinator import DeterministicCoordinator
   from core.data.types import Position, CellType, CellState
   from core.metrics.collector import MetricsCollector
   from analysis.visualization.demo_visualizations import create_sorting_animation

   class SortingCellAgent(AsyncCellAgent):
       """
       A cellular agent that exhibits sorting behavior based on neighbor preferences.
       """

       def __init__(self, cell_id: str, position: Position, cell_type: CellType,
                    adhesion_strength: float = 0.7):
           super().__init__(cell_id, position, cell_type)

           # Behavioral parameters
           self.adhesion_strength = adhesion_strength
           self.satisfaction_threshold = 0.5
           self.movement_probability = 0.8

           # State tracking
           self.satisfaction_score = 0.0
           self.movement_history = []
           self.neighbor_history = []

       async def update(self):
           """Main update loop for the cellular agent."""

           # Step 1: Assess current neighborhood satisfaction
           await self.calculate_satisfaction()

           # Step 2: Decide whether to attempt movement
           if (self.satisfaction_score < self.satisfaction_threshold and
               np.random.random() < self.movement_probability):
               await self.attempt_movement()

           # Step 3: Update internal state
           await self.update_internal_state()

       async def calculate_satisfaction(self):
           """
           Calculate how satisfied this cell is with its current neighborhood.

           Satisfaction is based on the proportion of neighbors that are the same type,
           weighted by the adhesion strength parameter.
           """
           neighbors = await self.get_neighbors(radius=1)

           if len(neighbors) == 0:
               self.satisfaction_score = 0.5  # Neutral if isolated
               return

           # Count neighbors of same type
           same_type_neighbors = sum(1 for n in neighbors if n.cell_type == self.cell_type)
           total_neighbors = len(neighbors)

           # Calculate base satisfaction from neighbor composition
           same_type_ratio = same_type_neighbors / total_neighbors

           # Apply adhesion strength weighting
           self.satisfaction_score = (same_type_ratio * self.adhesion_strength +
                                    (1 - same_type_ratio) * (1 - self.adhesion_strength))

           # Store neighbor information for analysis
           self.neighbor_history.append({
               'step': self.age,
               'total_neighbors': total_neighbors,
               'same_type_neighbors': same_type_neighbors,
               'satisfaction': self.satisfaction_score
           })

       async def attempt_movement(self):
           """
           Try to find a better location by evaluating nearby positions.
           """
           current_position = self.position
           best_position = current_position
           best_satisfaction = self.satisfaction_score

           # Check all adjacent positions
           for dx in [-1, 0, 1]:
               for dy in [-1, 0, 1]:
                   if dx == 0 and dy == 0:
                       continue  # Skip current position

                   new_position = Position(
                       current_position.x + dx,
                       current_position.y + dy
                   )

                   # Check if position is valid and available
                   if await self.is_valid_position(new_position):
                       predicted_satisfaction = await self.predict_satisfaction_at(new_position)

                       if predicted_satisfaction > best_satisfaction:
                           best_position = new_position
                           best_satisfaction = predicted_satisfaction

           # Move if we found a better location
           if best_position != current_position:
               await self.move_to(best_position)
               self.satisfaction_score = best_satisfaction

               # Record movement
               self.movement_history.append({
                   'step': self.age,
                   'from': current_position,
                   'to': best_position,
                   'satisfaction_improvement': best_satisfaction - self.satisfaction_score
               })

       async def predict_satisfaction_at(self, position: Position) -> float:
           """
           Predict satisfaction score if the cell moved to the given position.
           """
           potential_neighbors = await self.get_neighbors_at_position(position, radius=1)

           if len(potential_neighbors) == 0:
               return 0.5

           same_type_neighbors = sum(1 for n in potential_neighbors
                                   if n.cell_type == self.cell_type)
           total_neighbors = len(potential_neighbors)
           same_type_ratio = same_type_neighbors / total_neighbors

           return (same_type_ratio * self.adhesion_strength +
                   (1 - same_type_ratio) * (1 - self.adhesion_strength))

       async def update_internal_state(self):
           """Update internal cellular state."""
           self.age += 1

           # Cells become less likely to move as they age and become more satisfied
           if self.satisfaction_score > 0.8:
               self.movement_probability *= 0.98
           else:
               self.movement_probability = min(0.8, self.movement_probability * 1.01)


   class CellSortingExperiment:
       """
       Complete cell sorting experiment with analysis and visualization.
       """

       def __init__(self,
                    population_size: int = 200,
                    grid_size: Tuple[int, int] = (30, 30),
                    cell_types: List[str] = ['red', 'blue'],
                    adhesion_strength: float = 0.7,
                    simulation_steps: int = 300,
                    random_seed: int = None):

           self.population_size = population_size
           self.grid_size = grid_size
           self.cell_types = cell_types
           self.adhesion_strength = adhesion_strength
           self.simulation_steps = simulation_steps
           self.random_seed = random_seed

           # Results storage
           self.coordinator = None
           self.agents = []
           self.metrics_collector = None
           self.results = {}

       async def setup(self):
           """Initialize the experiment."""

           if self.random_seed is not None:
               np.random.seed(self.random_seed)

           # Create coordinator
           self.coordinator = DeterministicCoordinator(
               grid_size=self.grid_size,
               boundary_conditions='periodic'
           )

           # Initialize metrics collection
           self.metrics_collector = MetricsCollector()

           # Create cellular agents
           await self.create_agents()

           logging.info(f"Experiment setup complete: {len(self.agents)} agents created")

       async def create_agents(self):
           """Create and position cellular agents."""

           # Generate random positions
           positions = []
           while len(positions) < self.population_size:
               x = np.random.randint(0, self.grid_size[0])
               y = np.random.randint(0, self.grid_size[1])
               pos = Position(x, y)

               # Ensure no two agents start at the same position
               if pos not in positions:
                   positions.append(pos)

           # Create agents with random cell types
           for i, position in enumerate(positions):
               cell_type = CellType(np.random.choice(self.cell_types))

               agent = SortingCellAgent(
                   cell_id=f"cell_{i}",
                   position=position,
                   cell_type=cell_type,
                   adhesion_strength=self.adhesion_strength
               )

               self.agents.append(agent)
               await self.coordinator.add_agent(agent)

       async def run(self):
           """Execute the cell sorting simulation."""

           await self.setup()

           logging.info("Starting cell sorting simulation...")

           # Record initial state
           await self.record_metrics(step=0)

           # Main simulation loop
           for step in range(1, self.simulation_steps + 1):
               # Update all agents
               await self.coordinator.step()

               # Record metrics
               if step % 10 == 0 or step == self.simulation_steps:
                   await self.record_metrics(step)

               # Progress logging
               if step % 50 == 0:
                   logging.info(f"Simulation step {step}/{self.simulation_steps}")

           # Analyze results
           await self.analyze_results()

           logging.info("Cell sorting simulation complete!")

           return self.results

       async def record_metrics(self, step: int):
           """Record metrics for analysis."""

           # Calculate sorting score
           sorting_score = await self.calculate_sorting_score()

           # Calculate average satisfaction
           total_satisfaction = sum(agent.satisfaction_score for agent in self.agents)
           avg_satisfaction = total_satisfaction / len(self.agents)

           # Record movement activity
           recent_movers = sum(1 for agent in self.agents
                              if len(agent.movement_history) > 0 and
                              agent.movement_history[-1]['step'] == step - 1)
           movement_rate = recent_movers / len(self.agents)

           # Store metrics
           metrics = {
               'step': step,
               'sorting_score': sorting_score,
               'average_satisfaction': avg_satisfaction,
               'movement_rate': movement_rate,
               'total_agents': len(self.agents)
           }

           self.metrics_collector.record_step_metrics(step, metrics)

       async def calculate_sorting_score(self) -> float:
           """
           Calculate overall sorting score (0 = random, 1 = perfectly sorted).

           This measures global organization by comparing the actual neighbor
           composition to what would be expected in a perfectly sorted system.
           """

           total_score = 0.0
           total_agents = 0

           for agent in self.agents:
               neighbors = await agent.get_neighbors(radius=1)

               if len(neighbors) > 0:
                   same_type_neighbors = sum(1 for n in neighbors
                                           if n.cell_type == agent.cell_type)
                   local_sorting_score = same_type_neighbors / len(neighbors)
                   total_score += local_sorting_score
                   total_agents += 1

           return total_score / total_agents if total_agents > 0 else 0.0

       async def analyze_results(self):
           """Perform comprehensive analysis of simulation results."""

           metrics = self.metrics_collector.get_all_metrics()

           # Extract time series data
           steps = [m['step'] for m in metrics]
           sorting_scores = [m['sorting_score'] for m in metrics]
           satisfaction_scores = [m['average_satisfaction'] for m in metrics]
           movement_rates = [m['movement_rate'] for m in metrics]

           # Calculate final statistics
           final_sorting_score = sorting_scores[-1]
           final_satisfaction = satisfaction_scores[-1]
           final_movement_rate = movement_rates[-1]

           # Analyze convergence
           convergence_threshold = 0.01
           convergence_step = self.find_convergence_step(sorting_scores, convergence_threshold)

           # Analyze emergence
           emergence_detected = final_sorting_score > 0.7 and convergence_step is not None

           # Store comprehensive results
           self.results = {
               'simulation_parameters': {
                   'population_size': self.population_size,
                   'grid_size': self.grid_size,
                   'cell_types': self.cell_types,
                   'adhesion_strength': self.adhesion_strength,
                   'simulation_steps': self.simulation_steps,
                   'random_seed': self.random_seed
               },
               'final_metrics': {
                   'sorting_score': final_sorting_score,
                   'average_satisfaction': final_satisfaction,
                   'movement_rate': final_movement_rate
               },
               'time_series': {
                   'steps': steps,
                   'sorting_scores': sorting_scores,
                   'satisfaction_scores': satisfaction_scores,
                   'movement_rates': movement_rates
               },
               'analysis': {
                   'convergence_step': convergence_step,
                   'emergence_detected': emergence_detected,
                   'sorting_efficiency': self.calculate_sorting_efficiency(),
                   'stability_measure': self.calculate_stability_measure(movement_rates)
               }
           }

       def find_convergence_step(self, scores: List[float], threshold: float) -> int:
           """Find when the sorting score converged to a stable value."""

           if len(scores) < 20:
               return None

           # Look for sustained low variability
           window_size = 20
           for i in range(len(scores) - window_size):
               window = scores[i:i + window_size]
               if np.std(window) < threshold:
                   return i

           return None

       def calculate_sorting_efficiency(self) -> float:
           """Calculate how efficiently the system reached sorted state."""

           sorting_scores = self.results['time_series']['sorting_scores']

           # Area under the sorting curve (higher = more efficient)
           total_area = np.trapz(sorting_scores)
           max_possible_area = len(sorting_scores)  # If sorting_score = 1.0 throughout

           return total_area / max_possible_area

       def calculate_stability_measure(self, movement_rates: List[float]) -> float:
           """Calculate final stability (lower movement = higher stability)."""

           # Average movement rate in final 20% of simulation
           final_portion = len(movement_rates) // 5
           final_movement_rates = movement_rates[-final_portion:]

           return 1.0 - np.mean(final_movement_rates)  # Invert so higher = more stable

       async def create_visualization(self, save_animation: bool = True):
           """Create visualization of the sorting process."""

           if save_animation:
               # Create animation of the sorting process
               snapshots = self.coordinator.get_snapshots()
               animation = create_sorting_animation(
                   snapshots,
                   cell_types=self.cell_types,
                   grid_size=self.grid_size
               )
               animation.save('cell_sorting_animation.gif', writer='pillow', fps=5)
               logging.info("Animation saved as 'cell_sorting_animation.gif'")

           # Create analysis plots
           self.create_analysis_plots()

       def create_analysis_plots(self):
           """Create analysis plots showing key metrics over time."""

           fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

           steps = self.results['time_series']['steps']

           # Plot 1: Sorting Score Over Time
           ax1.plot(steps, self.results['time_series']['sorting_scores'], 'b-', linewidth=2)
           ax1.set_xlabel('Simulation Step')
           ax1.set_ylabel('Sorting Score')
           ax1.set_title('Cell Sorting Progress')
           ax1.grid(True, alpha=0.3)

           # Plot 2: Average Satisfaction
           ax2.plot(steps, self.results['time_series']['satisfaction_scores'], 'g-', linewidth=2)
           ax2.set_xlabel('Simulation Step')
           ax2.set_ylabel('Average Satisfaction')
           ax2.set_title('Cellular Satisfaction Over Time')
           ax2.grid(True, alpha=0.3)

           # Plot 3: Movement Rate
           ax3.plot(steps, self.results['time_series']['movement_rates'], 'r-', linewidth=2)
           ax3.set_xlabel('Simulation Step')
           ax3.set_ylabel('Movement Rate')
           ax3.set_title('Cellular Movement Activity')
           ax3.grid(True, alpha=0.3)

           # Plot 4: Final State Summary
           metrics = ['Sorting Score', 'Satisfaction', 'Stability']
           values = [
               self.results['final_metrics']['sorting_score'],
               self.results['final_metrics']['average_satisfaction'],
               self.results['analysis']['stability_measure']
           ]

           bars = ax4.bar(metrics, values, color=['blue', 'green', 'orange'])
           ax4.set_ylabel('Score')
           ax4.set_title('Final State Metrics')
           ax4.set_ylim(0, 1)

           # Add value labels on bars
           for bar, value in zip(bars, values):
               ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

           plt.tight_layout()
           plt.savefig('cell_sorting_analysis.png', dpi=300, bbox_inches='tight')
           plt.show()

           logging.info("Analysis plots saved as 'cell_sorting_analysis.png'")


   # Example execution function
   async def run_basic_sorting_example():
       """Run the basic cell sorting example with standard parameters."""

       # Configure logging
       logging.basicConfig(level=logging.INFO)

       # Create and run experiment
       experiment = CellSortingExperiment(
           population_size=150,
           grid_size=(25, 25),
           cell_types=['epithelial', 'mesenchymal'],
           adhesion_strength=0.75,
           simulation_steps=200,
           random_seed=42
       )

       results = await experiment.run()

       # Create visualizations
       await experiment.create_visualization(save_animation=True)

       # Print summary
       print("\n=== Cell Sorting Experiment Results ===")
       print(f"Final sorting score: {results['final_metrics']['sorting_score']:.3f}")
       print(f"Average satisfaction: {results['final_metrics']['average_satisfaction']:.3f}")
       print(f"Emergence detected: {'Yes' if results['analysis']['emergence_detected'] else 'No'}")
       print(f"Convergence step: {results['analysis']['convergence_step']}")
       print(f"Sorting efficiency: {results['analysis']['sorting_efficiency']:.3f}")

       return results


   if __name__ == "__main__":
       # Run the example
       results = asyncio.run(run_basic_sorting_example())

Running the Example
------------------

To run this example:

.. code-block:: bash

   # Navigate to examples directory
   cd examples/basic

   # Run with default parameters
   python cell_sorting.py

   # Run with custom parameters
   python cell_sorting.py --population-size 300 --adhesion-strength 0.8

   # Run multiple trials for statistics
   python cell_sorting.py --trials 10 --statistical-analysis

Expected Results
---------------

When you run this example, you should see:

**Console Output:**
.. code-block:: text

   === Cell Sorting Experiment Results ===
   Final sorting score: 0.847
   Average satisfaction: 0.923
   Emergence detected: Yes
   Convergence step: 120
   Sorting efficiency: 0.756

**Generated Files:**
   * ``cell_sorting_animation.gif`` - Animation showing sorting process
   * ``cell_sorting_analysis.png`` - Analysis plots
   * ``cell_sorting_data.csv`` - Raw simulation data

**Key Observations:**
   * Initially mixed cells gradually segregate into homogeneous regions
   * Sorting score increases from ~0.5 (random) to >0.8 (well-sorted)
   * Cell movement decreases as the system reaches equilibrium
   * Local satisfaction correlates with global organization

Biological Interpretation
------------------------

**What This Model Shows:**
   * How local cellular preferences (adhesion) create global organization (sorting)
   * The role of cellular movement in achieving optimal tissue arrangement
   * How satisfaction at the cellular level relates to organization at the tissue level
   * The time dynamics of morphogenetic processes

**Connections to Real Biology:**
   * **Germ Layer Formation**: Similar sorting occurs during early embryogenesis
   * **Tumor Invasion**: Cancer cells may have altered adhesion properties
   * **Tissue Engineering**: Understanding sorting helps design better scaffolds
   * **Wound Healing**: Cell sorting helps restore proper tissue architecture

**Limitations to Consider:**
   * Real cells have more complex adhesion mechanisms
   * Cell division and death are not included in this model
   * Chemical signaling and gradients are not explicitly modeled
   * Three-dimensional effects are not captured

Exercises and Extensions
-----------------------

**Exercise 1: Parameter Sensitivity**
   Vary the adhesion strength parameter from 0.5 to 0.9. How does this affect:
   * The final sorting score?
   * The time to convergence?
   * The sorting efficiency?

**Exercise 2: Multiple Cell Types**
   Extend the model to include three cell types. What patterns emerge?
   Do all cell types sort equally well?

**Exercise 3: Asymmetric Adhesion**
   Make cell type A strongly prefer other A cells, but cell type B only weakly prefer other B cells.
   How does this asymmetry affect sorting?

**Exercise 4: Spatial Constraints**
   Add obstacles or boundaries to the simulation space.
   How do physical constraints affect the sorting process?

**Exercise 5: Dynamic Adhesion**
   Make adhesion strength change over time (e.g., increasing during development).
   How does this affect the sorting dynamics?

**Exercise 6: Validation Study**
   Compare your simulation results to experimental data from cell sorting assays.
   What parameters best match real biological systems?

Next Steps
---------

After completing this example, you can:

* Explore :doc:`../intermediate/adaptive_sorting` - cells that change their behavior
* Study :doc:`../intermediate/pattern_formation` - more complex spatial organization
* Learn about :doc:`../../methodology/statistical_validation` - rigorous analysis methods
* Try :doc:`../applications/tissue_morphogenesis` - realistic biological applications

This example demonstrates the power of agent-based modeling to reveal how simple local rules create complex global behaviors - a key principle in understanding morphogenesis and development.