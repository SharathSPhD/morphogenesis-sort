Intermediate Examples
====================

This section provides intermediate-level examples that build upon the basic concepts while introducing more sophisticated features like multiple cell types, learning behaviors, and data analysis workflows.

Multi-Cell Type Sorting
------------------------

This example demonstrates cellular sorting with multiple interacting cell types and different adhesion preferences.

**Overview:**
Simulate the sorting of three different cell types with varying adhesion strengths, demonstrating how complex patterns can emerge from simple interaction rules.

**Implementation:**

.. code-block:: python

   import asyncio
   import numpy as np
   import matplotlib.pyplot as plt
   from core.coordination.coordinator import DeterministicCoordinator
   from core.agents.behaviors.sorting_cell import SortingCell
   from core.data.types import Position2D, CellType
   from analysis.visualization.comprehensive_visualization_suite import VisualizationSuite

   class MultiTypeSortingExperiment:
       def __init__(self, grid_size=(80, 80), population_composition=None):
           self.grid_size = grid_size
           self.coordinator = DeterministicCoordinator(
               grid_size=grid_size,
               max_agents=2000
           )

           # Default population composition
           self.population_composition = population_composition or {
               'type_A': {'count': 400, 'color': 'red', 'adhesion_strength': 1.0},
               'type_B': {'count': 400, 'color': 'blue', 'adhesion_strength': 0.8},
               'type_C': {'count': 200, 'color': 'green', 'adhesion_strength': 1.2}
           }

           # Differential adhesion matrix
           self.adhesion_matrix = self.create_adhesion_matrix()

       def create_adhesion_matrix(self):
           """Create differential adhesion matrix between cell types."""
           # Higher values = stronger adhesion
           adhesion_matrix = {
               'type_A': {'type_A': 1.0, 'type_B': 0.3, 'type_C': 0.5},
               'type_B': {'type_A': 0.3, 'type_B': 0.8, 'type_C': 0.2},
               'type_C': {'type_A': 0.5, 'type_B': 0.2, 'type_C': 1.2}
           }
           return adhesion_matrix

       async def initialize_mixed_population(self):
           """Initialize randomly mixed population of different cell types."""
           agents = []
           agent_id = 0

           for cell_type, properties in self.population_composition.items():
               for _ in range(properties['count']):
                   # Random initial position
                   position = Position2D(
                       x=np.random.randint(5, self.grid_size[0] - 5),
                       y=np.random.randint(5, self.grid_size[1] - 5)
                   )

                   # Create sorting cell with type-specific properties
                   agent = MultTypeSortingCell(
                       agent_id=agent_id,
                       initial_position=position,
                       cell_type=cell_type,
                       adhesion_matrix=self.adhesion_matrix,
                       adhesion_strength=properties['adhesion_strength']
                   )

                   agents.append(agent)
                   agent_id += 1

           await self.coordinator.add_agents(agents)
           return agents

       async def run_sorting_experiment(self, max_timesteps=3000, analysis_interval=100):
           """Run multi-type sorting experiment with analysis."""
           print("Starting multi-type cell sorting experiment...")

           # Initialize population
           agents = await self.initialize_mixed_population()

           # Analysis tracking
           sorting_metrics = []
           energy_history = []
           cluster_analysis = []

           # Main simulation loop
           for timestep in range(max_timesteps):
               await self.coordinator.step()

               # Periodic analysis
               if timestep % analysis_interval == 0:
                   print(f"Timestep {timestep}/{max_timesteps}")

                   # Calculate sorting quality
                   sorting_quality = await self.calculate_sorting_quality(agents)
                   sorting_metrics.append(sorting_quality)

                   # Calculate system energy
                   system_energy = await self.calculate_system_energy(agents)
                   energy_history.append(system_energy)

                   # Analyze cluster formation
                   cluster_data = await self.analyze_cluster_formation(agents)
                   cluster_analysis.append(cluster_data)

                   print(f"  Sorting quality: {sorting_quality:.3f}")
                   print(f"  System energy: {system_energy:.1f}")
                   print(f"  Largest clusters: {cluster_data['largest_clusters']}")

                   # Check convergence
                   if len(sorting_metrics) >= 5:
                       recent_change = abs(sorting_metrics[-1] - sorting_metrics[-5])
                       if recent_change < 0.01:
                           print(f"Converged at timestep {timestep}")
                           break

           return {
               'final_positions': [(a.position.x, a.position.y) for a in agents],
               'cell_types': [a.cell_type for a in agents],
               'sorting_metrics': sorting_metrics,
               'energy_history': energy_history,
               'cluster_analysis': cluster_analysis,
               'convergence_timestep': timestep
           }

       async def calculate_sorting_quality(self, agents):
           """Calculate overall sorting quality metric."""
           total_similar_neighbors = 0
           total_neighbors = 0

           for agent in agents:
               neighbors = await agent.get_neighbors(radius=3)

               for neighbor in neighbors:
                   total_neighbors += 1
                   if neighbor.cell_type == agent.cell_type:
                       total_similar_neighbors += 1

           return total_similar_neighbors / max(total_neighbors, 1)

       async def calculate_system_energy(self, agents):
           """Calculate total system energy based on adhesion interactions."""
           total_energy = 0

           for agent in agents:
               neighbors = await agent.get_neighbors(radius=2)
               for neighbor in neighbors:
                   # Get adhesion strength between cell types
                   adhesion = self.adhesion_matrix[agent.cell_type][neighbor.cell_type]
                   distance = np.linalg.norm(
                       np.array([agent.position.x - neighbor.position.x,
                                agent.position.y - neighbor.position.y])
                   )

                   # Energy decreases with stronger adhesion (more stable)
                   interaction_energy = -adhesion / max(distance, 0.1)
                   total_energy += interaction_energy

           return total_energy / 2  # Avoid double counting

       async def analyze_cluster_formation(self, agents):
           """Analyze cluster sizes and spatial organization."""
           # Group agents by type
           type_groups = {}
           for agent in agents:
               if agent.cell_type not in type_groups:
                   type_groups[agent.cell_type] = []
               type_groups[agent.cell_type].append(agent)

           cluster_data = {'clusters_by_type': {}, 'largest_clusters': {}}

           for cell_type, agents_of_type in type_groups.items():
               # Find connected components (clusters)
               clusters = await self.find_clusters(agents_of_type)

               cluster_sizes = [len(cluster) for cluster in clusters]
               largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0

               cluster_data['clusters_by_type'][cell_type] = {
                   'num_clusters': len(clusters),
                   'cluster_sizes': cluster_sizes,
                   'largest_cluster': largest_cluster_size,
                   'fragmentation_index': len(clusters) / len(agents_of_type)
               }

               cluster_data['largest_clusters'][cell_type] = largest_cluster_size

           return cluster_data

       async def find_clusters(self, agents_of_same_type):
           """Find connected clusters of same-type cells."""
           unvisited = set(agents_of_same_type)
           clusters = []

           while unvisited:
               # Start new cluster
               current_cluster = []
               to_visit = [unvisited.pop()]

               while to_visit:
                   current_agent = to_visit.pop()
                   current_cluster.append(current_agent)

                   # Find neighboring agents of same type
                   neighbors = await current_agent.get_neighbors(radius=2)
                   same_type_neighbors = [n for n in neighbors
                                        if n.cell_type == current_agent.cell_type
                                        and n in unvisited]

                   for neighbor in same_type_neighbors:
                       unvisited.remove(neighbor)
                       to_visit.append(neighbor)

               clusters.append(current_cluster)

           return clusters

       async def create_sorting_visualization(self, results):
           """Create comprehensive visualization of sorting results."""
           viz = VisualizationSuite()

           # Create figure with subplots
           fig, axes = plt.subplots(2, 3, figsize=(15, 10))

           # Final positions colored by type
           positions = results['final_positions']
           types = results['cell_types']

           type_colors = {
               'type_A': 'red',
               'type_B': 'blue',
               'type_C': 'green'
           }

           for cell_type in type_colors:
               type_positions = [pos for pos, t in zip(positions, types) if t == cell_type]
               if type_positions:
                   x_coords, y_coords = zip(*type_positions)
                   axes[0, 0].scatter(x_coords, y_coords,
                                    c=type_colors[cell_type],
                                    label=cell_type, alpha=0.7)

           axes[0, 0].set_title('Final Cell Positions')
           axes[0, 0].legend()
           axes[0, 0].set_xlim(0, self.grid_size[0])
           axes[0, 0].set_ylim(0, self.grid_size[1])

           # Sorting quality over time
           timesteps = np.arange(len(results['sorting_metrics']))
           axes[0, 1].plot(timesteps, results['sorting_metrics'])
           axes[0, 1].set_title('Sorting Quality Over Time')
           axes[0, 1].set_xlabel('Analysis Points')
           axes[0, 1].set_ylabel('Sorting Quality')

           # System energy over time
           axes[0, 2].plot(timesteps, results['energy_history'])
           axes[0, 2].set_title('System Energy Over Time')
           axes[0, 2].set_xlabel('Analysis Points')
           axes[0, 2].set_ylabel('Total Energy')

           # Cluster size evolution
           for cell_type in type_colors:
               cluster_sizes = [analysis['largest_clusters'][cell_type]
                              for analysis in results['cluster_analysis']]
               axes[1, 0].plot(timesteps, cluster_sizes,
                             color=type_colors[cell_type], label=cell_type)

           axes[1, 0].set_title('Largest Cluster Size Evolution')
           axes[1, 0].set_xlabel('Analysis Points')
           axes[1, 0].set_ylabel('Largest Cluster Size')
           axes[1, 0].legend()

           # Final cluster size distribution
           final_analysis = results['cluster_analysis'][-1]
           cell_types = list(type_colors.keys())
           cluster_counts = [final_analysis['clusters_by_type'][ct]['num_clusters']
                           for ct in cell_types]

           axes[1, 1].bar(cell_types, cluster_counts,
                         color=[type_colors[ct] for ct in cell_types])
           axes[1, 1].set_title('Final Number of Clusters')
           axes[1, 1].set_ylabel('Number of Clusters')

           # Fragmentation index
           fragmentation_indices = [final_analysis['clusters_by_type'][ct]['fragmentation_index']
                                  for ct in cell_types]

           axes[1, 2].bar(cell_types, fragmentation_indices,
                         color=[type_colors[ct] for ct in cell_types])
           axes[1, 2].set_title('Fragmentation Index')
           axes[1, 2].set_ylabel('Fragmentation Index')

           plt.tight_layout()
           plt.savefig('multi_type_sorting_analysis.png', dpi=300)
           plt.show()

           return fig

   class MultTypeSortingCell(SortingCell):
       def __init__(self, agent_id, initial_position, cell_type, adhesion_matrix, adhesion_strength):
           super().__init__(agent_id, initial_position, cell_type)
           self.adhesion_matrix = adhesion_matrix
           self.adhesion_strength = adhesion_strength

       async def calculate_local_adhesion_energy(self):
           """Calculate local adhesion energy with neighbors."""
           neighbors = await self.get_neighbors(radius=2)
           total_energy = 0

           for neighbor in neighbors:
               # Get adhesion coefficient between types
               adhesion_coeff = self.adhesion_matrix[self.cell_type][neighbor.cell_type]

               # Calculate distance-dependent energy
               distance = self.distance_to(neighbor)
               interaction_energy = adhesion_coeff / max(distance, 0.1)

               total_energy += interaction_energy

           return total_energy

       async def decide_movement(self):
           """Decide movement based on adhesion energy minimization."""
           current_energy = await self.calculate_local_adhesion_energy()

           best_position = self.position
           best_energy = current_energy

           # Test potential moves
           potential_moves = [
               (1, 0), (-1, 0), (0, 1), (0, -1),  # Adjacent
               (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
           ]

           for dx, dy in potential_moves:
               test_position = Position2D(
                   x=self.position.x + dx,
                   y=self.position.y + dy
               )

               # Check if move is valid
               if await self.is_valid_position(test_position):
                   # Calculate energy at test position
                   test_energy = await self.calculate_energy_at_position(test_position)

                   if test_energy > best_energy:  # Higher adhesion energy is better
                       best_energy = test_energy
                       best_position = test_position

           # Move if better position found
           if best_position != self.position:
               await self.move_to_position(best_position)

   # Example usage
   async def run_multi_type_sorting_example():
       # Create experiment with custom population
       experiment = MultiTypeSortingExperiment(
           grid_size=(100, 100),
           population_composition={
               'type_A': {'count': 500, 'color': 'red', 'adhesion_strength': 1.2},
               'type_B': {'count': 400, 'color': 'blue', 'adhesion_strength': 0.9},
               'type_C': {'count': 300, 'color': 'green', 'adhesion_strength': 1.1}
           }
       )

       # Run sorting experiment
       results = await experiment.run_sorting_experiment(
           max_timesteps=5000,
           analysis_interval=100
       )

       # Create visualization
       visualization = await experiment.create_sorting_visualization(results)

       print(f"\nExperiment Results:")
       print(f"Convergence timestep: {results['convergence_timestep']}")
       print(f"Final sorting quality: {results['sorting_metrics'][-1]:.3f}")
       print(f"Final system energy: {results['energy_history'][-1]:.1f}")

       return results

Adaptive Cell Behavior Learning
-------------------------------

This example demonstrates cells that learn and adapt their behavior based on environmental feedback.

**Overview:**
Implement cells that use reinforcement learning to adapt their movement and interaction strategies over time.

**Implementation:**

.. code-block:: python

   from core.agents.behaviors.adaptive_cell import AdaptiveCell
   from core.coordination.coordinator import DeterministicCoordinator
   import random

   class LearningCell(AdaptiveCell):
       def __init__(self, agent_id, initial_position, learning_rate=0.1):
           super().__init__(agent_id, initial_position)
           self.learning_rate = learning_rate

           # Q-learning parameters
           self.q_table = {}
           self.epsilon = 0.1  # Exploration rate
           self.gamma = 0.9    # Discount factor

           # Action space
           self.actions = [
               'move_north', 'move_south', 'move_east', 'move_west',
               'stay', 'communicate', 'reproduce', 'cooperate'
           ]

           # State representation
           self.previous_state = None
           self.previous_action = None

       async def perceive_environment(self):
           """Perceive current state of environment."""
           neighbors = await self.get_neighbors(radius=3)
           local_density = len(neighbors)

           # Categorize neighbors by type
           neighbor_types = {}
           for neighbor in neighbors:
               ntype = getattr(neighbor, 'cell_type', 'unknown')
               neighbor_types[ntype] = neighbor_types.get(ntype, 0) + 1

           # Resource availability (simulated)
           resource_level = await self.sense_local_resources()

           # Create state representation
           state = {
               'density': self.discretize_density(local_density),
               'dominant_neighbor_type': max(neighbor_types.items(), key=lambda x: x[1])[0] if neighbor_types else 'none',
               'resource_level': self.discretize_resource(resource_level),
               'energy_level': self.discretize_energy(self.energy)
           }

           return self.state_to_string(state)

       def state_to_string(self, state_dict):
           """Convert state dictionary to string for Q-table indexing."""
           return f"{state_dict['density']}_{state_dict['dominant_neighbor_type']}_" + \
                  f"{state_dict['resource_level']}_{state_dict['energy_level']}"

       def discretize_density(self, density):
           """Discretize density into categories."""
           if density == 0:
               return 'empty'
           elif density <= 2:
               return 'sparse'
           elif density <= 5:
               return 'medium'
           else:
               return 'crowded'

       def discretize_resource(self, resource_level):
           """Discretize resource level."""
           if resource_level < 0.3:
               return 'low'
           elif resource_level < 0.7:
               return 'medium'
           else:
               return 'high'

       def discretize_energy(self, energy_level):
           """Discretize energy level."""
           if energy_level < 0.3:
               return 'low'
           elif energy_level < 0.7:
               return 'medium'
           else:
               return 'high'

       def get_q_value(self, state, action):
           """Get Q-value for state-action pair."""
           if state not in self.q_table:
               self.q_table[state] = {action: 0.0 for action in self.actions}
           return self.q_table[state].get(action, 0.0)

       def choose_action(self, state):
           """Choose action using epsilon-greedy policy."""
           if random.random() < self.epsilon:
               # Exploration: random action
               return random.choice(self.actions)
           else:
               # Exploitation: best action
               q_values = {action: self.get_q_value(state, action) for action in self.actions}
               return max(q_values.keys(), key=lambda a: q_values[a])

       async def execute_action(self, action):
           """Execute chosen action."""
           if action == 'move_north':
               await self.move((0, 1))
           elif action == 'move_south':
               await self.move((0, -1))
           elif action == 'move_east':
               await self.move((1, 0))
           elif action == 'move_west':
               await self.move((-1, 0))
           elif action == 'stay':
               pass  # No movement
           elif action == 'communicate':
               await self.broadcast_signal('cooperation_request')
           elif action == 'reproduce':
               await self.attempt_reproduction()
           elif action == 'cooperate':
               await self.cooperate_with_neighbors()

       def calculate_reward(self, old_state, action, new_state):
           """Calculate reward based on state transition."""
           reward = 0.0

           # Reward for maintaining good energy
           if 'high' in new_state:
               reward += 0.1
           elif 'low' in new_state:
               reward -= 0.2

           # Reward for appropriate density behavior
           if 'crowded' in old_state and action.startswith('move'):
               reward += 0.05  # Good to move when crowded
           elif 'empty' in old_state and action == 'stay':
               reward -= 0.05  # Bad to stay when isolated

           # Reward cooperation in medium density
           if 'medium' in old_state and action in ['communicate', 'cooperate']:
               reward += 0.1

           return reward

       async def learning_step(self):
           """Perform one learning step."""
           # Perceive current state
           current_state = await self.perceive_environment()

           # Choose action
           action = self.choose_action(current_state)

           # Execute action
           await self.execute_action(action)

           # Update Q-table if we have previous experience
           if self.previous_state is not None and self.previous_action is not None:
               reward = self.calculate_reward(self.previous_state, self.previous_action, current_state)

               # Q-learning update
               old_q = self.get_q_value(self.previous_state, self.previous_action)
               best_future_q = max([self.get_q_value(current_state, a) for a in self.actions])

               new_q = old_q + self.learning_rate * (reward + self.gamma * best_future_q - old_q)

               # Initialize Q-table entry if needed
               if self.previous_state not in self.q_table:
                   self.q_table[self.previous_state] = {}
               self.q_table[self.previous_state][self.previous_action] = new_q

           # Store current state and action for next iteration
           self.previous_state = current_state
           self.previous_action = action

       def get_learning_statistics(self):
           """Get statistics about learning progress."""
           stats = {
               'q_table_size': len(self.q_table),
               'states_explored': len(self.q_table),
               'total_q_entries': sum(len(actions) for actions in self.q_table.values()),
               'exploration_rate': self.epsilon
           }

           # Calculate average Q-values
           all_q_values = []
           for state_actions in self.q_table.values():
               all_q_values.extend(state_actions.values())

           if all_q_values:
               stats['average_q_value'] = np.mean(all_q_values)
               stats['max_q_value'] = max(all_q_values)
               stats['min_q_value'] = min(all_q_values)

           return stats

   class AdaptiveBehaviorExperiment:
       def __init__(self, grid_size=(60, 60), num_agents=100):
           self.grid_size = grid_size
           self.num_agents = num_agents
           self.coordinator = DeterministicCoordinator(
               grid_size=grid_size,
               max_agents=num_agents
           )

       async def initialize_learning_agents(self):
           """Initialize population of learning agents."""
           agents = []

           for agent_id in range(self.num_agents):
               position = Position2D(
                   x=np.random.randint(0, self.grid_size[0]),
                   y=np.random.randint(0, self.grid_size[1])
               )

               agent = LearningCell(
                   agent_id=agent_id,
                   initial_position=position,
                   learning_rate=0.1
               )

               # Initialize random energy
               agent.energy = np.random.uniform(0.3, 1.0)

               agents.append(agent)

           await self.coordinator.add_agents(agents)
           return agents

       async def run_learning_experiment(self, num_episodes=1000, episode_length=200):
           """Run learning experiment with multiple episodes."""
           print(f"Starting adaptive behavior learning experiment...")
           print(f"Episodes: {num_episodes}, Episode length: {episode_length}")

           # Initialize agents
           agents = await self.initialize_learning_agents()

           # Tracking variables
           episode_rewards = []
           learning_progress = []
           convergence_metrics = []

           for episode in range(num_episodes):
               episode_reward = 0
               episode_actions = {action: 0 for action in agents[0].actions}

               # Run episode
               for step in range(episode_length):
                   # All agents take learning steps
                   for agent in agents:
                       await agent.learning_step()

                   # Update environment (resources, etc.)
                   await self.update_environment()

               # Analyze episode
               total_reward = sum(agent.calculate_episode_reward() for agent in agents)
               episode_rewards.append(total_reward)

               # Collect learning statistics
               if episode % 50 == 0:
                   learning_stats = self.analyze_learning_progress(agents)
                   learning_progress.append(learning_stats)

                   print(f"Episode {episode}: Avg reward = {total_reward/len(agents):.2f}, "
                         f"Exploration rate = {agents[0].epsilon:.3f}")

                   # Decay exploration rate
                   for agent in agents:
                       agent.epsilon = max(0.01, agent.epsilon * 0.995)

               # Check convergence
               if episode >= 100:
                   recent_rewards = episode_rewards[-50:]
                   convergence = np.std(recent_rewards) < 0.1 * np.mean(recent_rewards)
                   convergence_metrics.append(convergence)

                   if convergence and episode > 500:
                       print(f"Learning converged at episode {episode}")
                       break

           return {
               'episode_rewards': episode_rewards,
               'learning_progress': learning_progress,
               'final_q_tables': [agent.q_table for agent in agents],
               'convergence_episode': episode
           }

       def analyze_learning_progress(self, agents):
           """Analyze learning progress across all agents."""
           all_stats = [agent.get_learning_statistics() for agent in agents]

           aggregated_stats = {
               'avg_q_table_size': np.mean([s['q_table_size'] for s in all_stats]),
               'avg_states_explored': np.mean([s['states_explored'] for s in all_stats]),
               'avg_q_value': np.mean([s.get('average_q_value', 0) for s in all_stats]),
               'max_q_value': max([s.get('max_q_value', 0) for s in all_stats]),
               'exploration_rate': all_stats[0]['exploration_rate']
           }

           return aggregated_stats

       async def update_environment(self):
           """Update environmental conditions."""
           # Simulate resource regeneration/depletion
           # This is a placeholder for more complex environmental dynamics
           pass

   # Example usage
   async def run_adaptive_learning_example():
       experiment = AdaptiveBehaviorExperiment(
           grid_size=(80, 80),
           num_agents=50
       )

       results = await experiment.run_learning_experiment(
           num_episodes=800,
           episode_length=150
       )

       # Analyze results
       print(f"\nLearning Results:")
       print(f"Convergence episode: {results['convergence_episode']}")
       print(f"Final average reward: {results['episode_rewards'][-1]/50:.2f}")

       # Plot learning curves
       plt.figure(figsize=(12, 8))

       # Episode rewards
       plt.subplot(2, 2, 1)
       plt.plot(results['episode_rewards'])
       plt.title('Episode Rewards Over Time')
       plt.xlabel('Episode')
       plt.ylabel('Total Reward')

       # Q-table size growth
       if results['learning_progress']:
           episodes = np.arange(0, len(results['episode_rewards']), 50)[:len(results['learning_progress'])]
           q_table_sizes = [lp['avg_q_table_size'] for lp in results['learning_progress']]

           plt.subplot(2, 2, 2)
           plt.plot(episodes, q_table_sizes)
           plt.title('Average Q-Table Size Growth')
           plt.xlabel('Episode')
           plt.ylabel('Average Q-Table Size')

           # Average Q-values
           avg_q_values = [lp['avg_q_value'] for lp in results['learning_progress']]

           plt.subplot(2, 2, 3)
           plt.plot(episodes, avg_q_values)
           plt.title('Average Q-Value Evolution')
           plt.xlabel('Episode')
           plt.ylabel('Average Q-Value')

           # Exploration rate decay
           exploration_rates = [lp['exploration_rate'] for lp in results['learning_progress']]

           plt.subplot(2, 2, 4)
           plt.plot(episodes, exploration_rates)
           plt.title('Exploration Rate Decay')
           plt.xlabel('Episode')
           plt.ylabel('Exploration Rate')

       plt.tight_layout()
       plt.savefig('adaptive_learning_results.png', dpi=300)
       plt.show()

       return results

Morphogen Gradient Following
-----------------------------

This example demonstrates cells following chemical gradients to form organized patterns.

**Overview:**
Create a system where cells follow morphogen gradients to self-organize into spatial patterns, similar to biological development processes.

**Implementation:**

.. code-block:: python

   from core.agents.behaviors.morphogen_cell import MorphogenCell
   import scipy.ndimage as ndimage

   class MorphogenGradientExperiment:
       def __init__(self, domain_size=(100, 100)):
           self.domain_size = domain_size
           self.dx = 1.0  # Spatial resolution

           # Morphogen fields
           self.morphogen_fields = {}
           self.initialize_morphogen_fields()

           # Cellular agents
           self.coordinator = DeterministicCoordinator(
               grid_size=domain_size,
               max_agents=1000
           )

       def initialize_morphogen_fields(self):
           """Initialize morphogen fields with different patterns."""
           # Primary morphogen - forms gradient from source
           primary_field = np.zeros(self.domain_size)
           # Set source at one corner
           primary_field[10:20, 10:20] = 1.0
           self.morphogen_fields['primary'] = primary_field

           # Secondary morphogen - forms opposing gradient
           secondary_field = np.zeros(self.domain_size)
           # Set source at opposite corner
           secondary_field[-20:-10, -20:-10] = 1.0
           self.morphogen_fields['secondary'] = secondary_field

           # Inhibitor morphogen - forms central pattern
           inhibitor_field = np.zeros(self.domain_size)
           center_x, center_y = self.domain_size[0]//2, self.domain_size[1]//2
           inhibitor_field[center_x-5:center_x+5, center_y-5:center_y+5] = 0.8
           self.morphogen_fields['inhibitor'] = inhibitor_field

       async def diffuse_morphogens(self, dt=0.1):
           """Update morphogen fields through diffusion."""
           for name, field in self.morphogen_fields.items():
               # Diffusion parameters
               if name == 'primary':
                   diffusion_coeff = 0.1
                   decay_rate = 0.01
               elif name == 'secondary':
                   diffusion_coeff = 0.15
                   decay_rate = 0.01
               elif name == 'inhibitor':
                   diffusion_coeff = 0.05
                   decay_rate = 0.005

               # Apply diffusion (Gaussian blur approximation)
               diffused = ndimage.gaussian_filter(field, sigma=diffusion_coeff*dt*10)

               # Apply decay
               decayed = diffused * (1 - decay_rate * dt)

               # Update field
               self.morphogen_fields[name] = decayed

               # Maintain sources
               if name == 'primary':
                   self.morphogen_fields[name][10:20, 10:20] = 1.0
               elif name == 'secondary':
                   self.morphogen_fields[name][-20:-10, -20:-10] = 1.0
               elif name == 'inhibitor':
                   center_x, center_y = self.domain_size[0]//2, self.domain_size[1]//2
                   self.morphogen_fields[name][center_x-5:center_x+5, center_y-5:center_y+5] = 0.8

       async def initialize_responsive_cells(self, num_cells=200):
           """Initialize cells that respond to morphogen gradients."""
           agents = []

           for agent_id in range(num_cells):
               # Random initial position
               position = Position2D(
                   x=np.random.randint(5, self.domain_size[0]-5),
                   y=np.random.randint(5, self.domain_size[1]-5)
               )

               # Create different cell types with different sensitivities
               cell_type = f"type_{agent_id % 3}"
               agent = GradientFollowingCell(
                   agent_id=agent_id,
                   initial_position=position,
                   cell_type=cell_type,
                   morphogen_fields=self.morphogen_fields
               )

               agents.append(agent)

           await self.coordinator.add_agents(agents)
           return agents

       async def run_pattern_formation_experiment(self, simulation_time=500):
           """Run pattern formation experiment."""
           print("Starting morphogen gradient pattern formation experiment...")

           # Initialize cells
           agents = await self.initialize_responsive_cells()

           # Tracking
           pattern_metrics = []
           field_evolution = []

           for timestep in range(simulation_time):
               # Update morphogen fields
               await self.diffuse_morphogens()

               # Update cell behaviors
               await self.coordinator.step()

               # Record data every 10 timesteps
               if timestep % 10 == 0:
                   pattern_data = await self.analyze_spatial_pattern(agents)
                   pattern_metrics.append(pattern_data)

                   # Store field snapshots occasionally
                   if timestep % 50 == 0:
                       field_snapshot = {name: field.copy()
                                       for name, field in self.morphogen_fields.items()}
                       field_evolution.append(field_snapshot)

                       print(f"Timestep {timestep}: Pattern complexity = {pattern_data['complexity']:.3f}")

           return {
               'final_positions': [(a.position.x, a.position.y) for a in agents],
               'cell_types': [a.cell_type for a in agents],
               'pattern_metrics': pattern_metrics,
               'field_evolution': field_evolution,
               'final_fields': self.morphogen_fields.copy()
           }

       async def analyze_spatial_pattern(self, agents):
           """Analyze spatial organization of cells."""
           # Group cells by type
           type_positions = {}
           for agent in agents:
               if agent.cell_type not in type_positions:
                   type_positions[agent.cell_type] = []
               type_positions[agent.cell_type].append((agent.position.x, agent.position.y))

           # Calculate spatial metrics
           pattern_complexity = 0
           spatial_organization = {}

           for cell_type, positions in type_positions.items():
               if len(positions) < 2:
                   continue

               # Calculate spatial clustering
               positions_array = np.array(positions)
               center_of_mass = np.mean(positions_array, axis=0)
               spread = np.std(positions_array, axis=0)

               spatial_organization[cell_type] = {
                   'center_of_mass': center_of_mass,
                   'spread': spread,
                   'cell_count': len(positions)
               }

               # Add to complexity measure
               pattern_complexity += np.linalg.norm(spread)

           return {
               'complexity': pattern_complexity,
               'spatial_organization': spatial_organization,
               'total_cells': len(agents)
           }

   class GradientFollowingCell(MorphogenCell):
       def __init__(self, agent_id, initial_position, cell_type, morphogen_fields):
           super().__init__(agent_id, initial_position, cell_type)
           self.morphogen_fields = morphogen_fields

           # Gradient sensitivity based on cell type
           if cell_type == 'type_0':
               self.sensitivities = {'primary': 1.0, 'secondary': -0.5, 'inhibitor': -0.8}
           elif cell_type == 'type_1':
               self.sensitivities = {'primary': -0.3, 'secondary': 1.2, 'inhibitor': -0.6}
           else:  # type_2
               self.sensitivities = {'primary': 0.2, 'secondary': 0.2, 'inhibitor': 1.0}

           self.movement_speed = 0.8

       async def sense_morphogen_gradients(self):
           """Sense morphogen gradients at current position."""
           gradients = {}

           for morphogen_name, field in self.morphogen_fields.items():
               gradient = self.calculate_gradient_at_position(field, self.position)
               gradients[morphogen_name] = gradient

           return gradients

       def calculate_gradient_at_position(self, field, position):
           """Calculate gradient at given position using finite differences."""
           x, y = int(position.x), int(position.y)

           # Bounds checking
           if x <= 0 or x >= field.shape[0]-1 or y <= 0 or y >= field.shape[1]-1:
               return np.array([0.0, 0.0])

           # Calculate gradient using central differences
           grad_x = (field[x+1, y] - field[x-1, y]) / (2 * self.dx)
           grad_y = (field[x, y+1] - field[x, y-1]) / (2 * self.dx)

           return np.array([grad_x, grad_y])

       async def calculate_movement_direction(self):
           """Calculate movement direction based on morphogen gradients."""
           gradients = await self.sense_morphogen_gradients()

           # Combine gradients based on sensitivities
           total_gradient = np.array([0.0, 0.0])

           for morphogen_name, gradient in gradients.items():
               sensitivity = self.sensitivities.get(morphogen_name, 0.0)
               total_gradient += sensitivity * gradient

           # Normalize and scale by movement speed
           if np.linalg.norm(total_gradient) > 0:
               direction = total_gradient / np.linalg.norm(total_gradient)
               return direction * self.movement_speed
           else:
               return np.array([0.0, 0.0])

       async def morphogen_response_step(self):
           """Execute morphogen response behavior."""
           # Calculate movement direction
           movement = await self.calculate_movement_direction()

           # Add some random noise to prevent getting stuck
           noise = np.random.normal(0, 0.1, 2)
           movement += noise

           # Execute movement
           new_position = Position2D(
               x=self.position.x + movement[0],
               y=self.position.y + movement[1]
           )

           # Check bounds
           new_position.x = np.clip(new_position.x, 0, self.morphogen_fields['primary'].shape[0]-1)
           new_position.y = np.clip(new_position.y, 0, self.morphogen_fields['primary'].shape[1]-1)

           await self.move_to_position(new_position)

   # Example usage
   async def run_morphogen_gradient_example():
       experiment = MorphogenGradientExperiment(domain_size=(120, 120))

       results = await experiment.run_pattern_formation_experiment(simulation_time=1000)

       # Create comprehensive visualization
       fig, axes = plt.subplots(2, 3, figsize=(18, 12))

       # Final morphogen fields
       morphogen_names = ['primary', 'secondary', 'inhibitor']
       for i, name in enumerate(morphogen_names):
           im = axes[0, i].imshow(results['final_fields'][name], cmap='viridis')
           axes[0, i].set_title(f'{name.title()} Morphogen Field')
           plt.colorbar(im, ax=axes[0, i])

       # Final cell positions
       positions = results['final_positions']
       types = results['cell_types']

       colors = {'type_0': 'red', 'type_1': 'blue', 'type_2': 'green'}
       for cell_type in colors:
           type_positions = [pos for pos, t in zip(positions, types) if t == cell_type]
           if type_positions:
               x_coords, y_coords = zip(*type_positions)
               axes[1, 0].scatter(x_coords, y_coords, c=colors[cell_type],
                                label=cell_type, alpha=0.7)

       axes[1, 0].set_title('Final Cell Positions')
       axes[1, 0].legend()

       # Pattern complexity over time
       timesteps = np.arange(len(results['pattern_metrics']))
       complexities = [pm['complexity'] for pm in results['pattern_metrics']]

       axes[1, 1].plot(timesteps, complexities)
       axes[1, 1].set_title('Pattern Complexity Over Time')
       axes[1, 1].set_xlabel('Time (×10 timesteps)')
       axes[1, 1].set_ylabel('Complexity')

       # Cell count by type over time
       for cell_type in colors:
           counts = []
           for pm in results['pattern_metrics']:
               if cell_type in pm['spatial_organization']:
                   counts.append(pm['spatial_organization'][cell_type]['cell_count'])
               else:
                   counts.append(0)
           axes[1, 2].plot(timesteps, counts, color=colors[cell_type], label=cell_type)

       axes[1, 2].set_title('Cell Count by Type')
       axes[1, 2].set_xlabel('Time (×10 timesteps)')
       axes[1, 2].set_ylabel('Cell Count')
       axes[1, 2].legend()

       plt.tight_layout()
       plt.savefig('morphogen_gradient_results.png', dpi=300)
       plt.show()

       return results

   # Run all examples
   async def run_all_intermediate_examples():
       print("Running intermediate examples...")

       print("\n" + "="*50)
       print("1. Multi-Type Cell Sorting")
       print("="*50)
       sorting_results = await run_multi_type_sorting_example()

       print("\n" + "="*50)
       print("2. Adaptive Learning Behavior")
       print("="*50)
       learning_results = await run_adaptive_learning_example()

       print("\n" + "="*50)
       print("3. Morphogen Gradient Following")
       print("="*50)
       gradient_results = await run_morphogen_gradient_example()

       return {
           'sorting': sorting_results,
           'learning': learning_results,
           'gradients': gradient_results
       }

   # Run examples
   if __name__ == "__main__":
       results = asyncio.run(run_all_intermediate_examples())

Conclusion
----------

These intermediate examples demonstrate several key concepts in morphogenesis simulations:

**Multi-Type Interactions:**
- Complex sorting patterns from differential adhesion
- Energy-based optimization of cellular arrangements
- Quantitative analysis of spatial organization

**Adaptive Learning:**
- Reinforcement learning in cellular agents
- Environmental perception and state representation
- Q-learning for behavior optimization

**Chemical Signaling:**
- Morphogen gradient formation and maintenance
- Cellular responses to chemical cues
- Pattern formation through gradient following

**Key Takeaways:**
- Simple local rules can create complex global patterns
- Learning mechanisms enable adaptive behavior
- Chemical gradients provide positional information
- Quantitative analysis reveals emergent properties

These examples bridge the gap between basic cellular sorting and advanced multi-scale modeling, providing the foundation for understanding more complex biological and engineering applications of morphogenetic principles.