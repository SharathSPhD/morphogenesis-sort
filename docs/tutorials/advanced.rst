Advanced Tutorials
==================

These tutorials cover cutting-edge techniques in morphogenesis research, including machine learning integration, distributed computing, and novel algorithmic approaches.

Tutorial 1: Machine Learning Integration
-----------------------------------------

Learn how to integrate machine learning techniques with morphogenesis simulations for pattern recognition, behavior prediction, and automated discovery.

**Learning Objectives:**
- Implement neural networks for cellular behavior control
- Use reinforcement learning for emergent strategy discovery
- Apply deep learning for pattern analysis and classification
- Create hybrid ML-morphogenesis systems

**Prerequisites:**
- Completed intermediate tutorials
- Familiarity with TensorFlow/PyTorch
- Understanding of machine learning concepts

Step 1: Neural Network-Controlled Cellular Agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, let's create cellular agents that use neural networks to make decisions.

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   import numpy as np
   from core.agents.cell_agent import AsyncCellAgent

   class NeuralCellBehaviorNetwork(nn.Module):
       def __init__(self, input_size=10, hidden_size=64, output_size=8):
           super(NeuralCellBehaviorNetwork, self).__init__()

           self.network = nn.Sequential(
               nn.Linear(input_size, hidden_size),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(hidden_size, hidden_size),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(hidden_size, output_size),
               nn.Softmax(dim=-1)
           )

       def forward(self, x):
           return self.network(x)

   class MLControlledCell(AsyncCellAgent):
       def __init__(self, agent_id, initial_position, network_path=None):
           super().__init__(agent_id, initial_position)

           # Neural network for behavior control
           self.behavior_network = NeuralCellBehaviorNetwork()
           if network_path:
               self.behavior_network.load_state_dict(torch.load(network_path))

           # Action space
           self.actions = [
               'move_north', 'move_south', 'move_east', 'move_west',
               'stay', 'communicate', 'divide', 'cooperate'
           ]

           # Experience buffer for learning
           self.experience_buffer = []
           self.max_buffer_size = 1000

       async def perceive_environment(self):
           """Create neural network input from environment perception."""
           # Get basic environmental information
           neighbors = await self.get_neighbors(radius=5)
           local_density = len(neighbors) / 25.0  # Normalize by max possible neighbors

           # Neighbor type distribution
           neighbor_types = {'type_A': 0, 'type_B': 0, 'type_C': 0}
           for neighbor in neighbors:
               ntype = getattr(neighbor, 'cell_type', 'type_A')
               neighbor_types[ntype] = neighbor_types.get(ntype, 0) + 1

           # Normalize neighbor counts
           total_neighbors = max(len(neighbors), 1)
           type_ratios = [neighbor_types[t] / total_neighbors for t in ['type_A', 'type_B', 'type_C']]

           # Resource and morphogen levels
           resource_level = await self.sense_local_resources()
           morphogen_levels = await self.sense_morphogens(['signal_A', 'signal_B'])

           # Internal state
           energy_level = getattr(self, 'energy', 0.5)
           age_normalized = min(getattr(self, 'age', 0) / 1000.0, 1.0)

           # Combine all features
           features = [
               local_density,
               *type_ratios,
               resource_level,
               *morphogen_levels.values(),
               energy_level,
               age_normalized
           ]

           # Pad or trim to expected input size
           while len(features) < 10:
               features.append(0.0)

           return torch.tensor(features[:10], dtype=torch.float32)

       async def neural_decision_step(self):
           """Use neural network to make behavioral decisions."""
           # Get environmental input
           perception_vector = await self.perceive_environment()

           # Forward pass through network
           with torch.no_grad():
               action_probabilities = self.behavior_network(perception_vector.unsqueeze(0))

           # Sample action based on probabilities
           action_idx = torch.multinomial(action_probabilities.squeeze(), 1).item()
           chosen_action = self.actions[action_idx]

           # Store experience for potential learning
           if len(self.experience_buffer) < self.max_buffer_size:
               self.experience_buffer.append({
                   'state': perception_vector.numpy(),
                   'action': action_idx,
                   'timestamp': len(self.experience_buffer)
               })

           return chosen_action

       async def execute_neural_action(self, action):
           """Execute the action chosen by the neural network."""
           if action == 'move_north':
               await self.move((0, 1))
           elif action == 'move_south':
               await self.move((0, -1))
           elif action == 'move_east':
               await self.move((1, 0))
           elif action == 'move_west':
               await self.move((-1, 0))
           elif action == 'stay':
               pass
           elif action == 'communicate':
               await self.broadcast_signal('neural_communication')
           elif action == 'divide':
               await self.attempt_reproduction()
           elif action == 'cooperate':
               await self.cooperate_with_neighbors()

       async def step(self, dt=0.1):
           """Main step function using neural network control."""
           action = await self.neural_decision_step()
           await self.execute_neural_action(action)
           await self.update_internal_state(dt)

Step 2: Reinforcement Learning for Emergent Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement a reinforcement learning system to discover optimal cellular strategies.

.. code-block:: python

   from collections import deque
   import random

   class MorphogenesisRLEnvironment:
       def __init__(self, grid_size=(50, 50), num_agents=100):
           self.grid_size = grid_size
           self.num_agents = num_agents
           self.coordinator = DeterministicCoordinator(
               grid_size=grid_size,
               max_agents=num_agents
           )

           # RL environment state
           self.current_step = 0
           self.max_steps = 1000
           self.agents = []

       async def reset(self):
           """Reset environment for new episode."""
           self.current_step = 0
           self.agents = []

           # Create RL-controlled agents
           for i in range(self.num_agents):
               agent = RLControlledCell(
                   agent_id=i,
                   initial_position=self.generate_random_position()
               )
               self.agents.append(agent)

           await self.coordinator.add_agents(self.agents)

           return await self.get_global_state()

       async def step(self, actions):
           """Execute one step of the environment."""
           # Apply actions to agents
           for agent, action in zip(self.agents, actions):
               await agent.execute_rl_action(action)

           # Update coordination
           await self.coordinator.step()

           # Calculate rewards
           rewards = await self.calculate_rewards()

           # Update environment state
           self.current_step += 1
           done = self.current_step >= self.max_steps

           next_state = await self.get_global_state()
           info = await self.get_environment_info()

           return next_state, rewards, done, info

       async def get_global_state(self):
           """Get global state representation for RL."""
           # Spatial density map
           density_map = np.zeros(self.grid_size)
           for agent in self.agents:
               x, y = int(agent.position.x), int(agent.position.y)
               if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                   density_map[x, y] += 1

           # Global metrics
           global_metrics = await self.calculate_global_metrics()

           # Combine spatial and global information
           state = {
               'density_map': density_map.flatten(),
               'global_metrics': list(global_metrics.values()),
               'timestep': self.current_step / self.max_steps
           }

           return state

       async def calculate_rewards(self):
           """Calculate rewards for RL agents."""
           rewards = []

           # Global objectives
           global_order = await self.calculate_global_order()
           global_efficiency = await self.calculate_global_efficiency()

           for agent in self.agents:
               reward = 0.0

               # Individual objectives
               local_neighbors = await agent.get_neighbors()
               same_type_neighbors = sum(1 for n in local_neighbors
                                       if n.cell_type == agent.cell_type)

               # Reward for local sorting
               if len(local_neighbors) > 0:
                   local_sorting = same_type_neighbors / len(local_neighbors)
                   reward += local_sorting * 0.3

               # Reward for contributing to global order
               reward += global_order * 0.5

               # Penalty for inefficient movements
               movement_efficiency = getattr(agent, 'movement_efficiency', 1.0)
               reward += movement_efficiency * 0.2

               # Small penalty for each timestep to encourage faster convergence
               reward -= 0.01

               rewards.append(reward)

           return rewards

   class RLControlledCell(MLControlledCell):
       def __init__(self, agent_id, initial_position):
           super().__init__(agent_id, initial_position)

           # RL-specific attributes
           self.q_network = self.create_q_network()
           self.target_network = self.create_q_network()
           self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

           # Experience replay
           self.replay_buffer = deque(maxlen=10000)
           self.batch_size = 32
           self.epsilon = 1.0
           self.epsilon_decay = 0.995
           self.epsilon_min = 0.01

       def create_q_network(self):
           """Create Q-network for deep Q-learning."""
           return nn.Sequential(
               nn.Linear(10, 128),  # Input size: environment features
               nn.ReLU(),
               nn.Linear(128, 128),
               nn.ReLU(),
               nn.Linear(128, len(self.actions))  # Output: Q-values for each action
           )

       async def choose_rl_action(self, state):
           """Choose action using epsilon-greedy policy."""
           if random.random() < self.epsilon:
               # Exploration: random action
               return random.randint(0, len(self.actions) - 1)
           else:
               # Exploitation: best action according to Q-network
               state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
               with torch.no_grad():
                   q_values = self.q_network(state_tensor)
               return torch.argmax(q_values).item()

       async def learn_from_experience(self):
           """Learn from stored experiences using DQN."""
           if len(self.replay_buffer) < self.batch_size:
               return

           # Sample batch from replay buffer
           batch = random.sample(self.replay_buffer, self.batch_size)

           states = torch.tensor([e['state'] for e in batch], dtype=torch.float32)
           actions = torch.tensor([e['action'] for e in batch], dtype=torch.long)
           rewards = torch.tensor([e['reward'] for e in batch], dtype=torch.float32)
           next_states = torch.tensor([e['next_state'] for e in batch], dtype=torch.float32)
           dones = torch.tensor([e['done'] for e in batch], dtype=torch.bool)

           # Current Q values
           current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

           # Next Q values from target network
           with torch.no_grad():
               next_q_values = self.target_network(next_states).max(1)[0]
               target_q_values = rewards + (0.99 * next_q_values * ~dones)

           # Loss calculation
           loss = nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)

           # Optimization step
           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()

           # Decay epsilon
           if self.epsilon > self.epsilon_min:
               self.epsilon *= self.epsilon_decay

       def store_experience(self, state, action, reward, next_state, done):
           """Store experience in replay buffer."""
           self.replay_buffer.append({
               'state': state,
               'action': action,
               'reward': reward,
               'next_state': next_state,
               'done': done
           })

Step 3: Deep Learning for Pattern Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create deep learning models for automatic pattern recognition and classification.

.. code-block:: python

   import torch.nn.functional as F
   from torch.utils.data import Dataset, DataLoader
   import torchvision.transforms as transforms

   class MorphogenesisPatternDataset(Dataset):
       def __init__(self, pattern_data, labels, transform=None):
           self.patterns = pattern_data  # Spatial patterns as 2D arrays
           self.labels = labels          # Pattern classifications
           self.transform = transform

       def __len__(self):
           return len(self.patterns)

       def __getitem__(self, idx):
           pattern = self.patterns[idx]
           label = self.labels[idx]

           if self.transform:
               pattern = self.transform(pattern)

           return torch.tensor(pattern, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

   class MorphogenesisPatternClassifier(nn.Module):
       def __init__(self, input_size=(50, 50), num_classes=5):
           super(MorphogenesisPatternClassifier, self).__init__()

           # Convolutional layers for spatial pattern recognition
           self.conv_layers = nn.Sequential(
               nn.Conv2d(1, 32, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),

               nn.Conv2d(32, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),

               nn.Conv2d(64, 128, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2)
           )

           # Calculate flattened size
           self.flattened_size = 128 * (input_size[0] // 8) * (input_size[1] // 8)

           # Fully connected layers
           self.fc_layers = nn.Sequential(
               nn.Linear(self.flattened_size, 256),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(256, 128),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(128, num_classes)
           )

       def forward(self, x):
           # Add channel dimension if needed
           if len(x.shape) == 3:
               x = x.unsqueeze(1)

           # Convolutional feature extraction
           features = self.conv_layers(x)

           # Flatten for fully connected layers
           features = features.view(features.size(0), -1)

           # Classification
           output = self.fc_layers(features)

           return output

   class PatternAnalysisSystem:
       def __init__(self):
           self.classifier = MorphogenesisPatternClassifier()
           self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
           self.criterion = nn.CrossEntropyLoss()

           # Pattern categories
           self.pattern_categories = [
               'random', 'clustered', 'striped', 'spotted', 'spiral'
           ]

       async def generate_training_data(self, num_samples=1000):
           """Generate training data from simulations."""
           patterns = []
           labels = []

           print("Generating training data from simulations...")

           for category_idx, category in enumerate(self.pattern_categories):
               print(f"Generating {category} patterns...")

               for sample in range(num_samples // len(self.pattern_categories)):
                   # Run simulation with parameters that produce this pattern type
                   pattern_data = await self.run_pattern_simulation(category)

                   patterns.append(pattern_data)
                   labels.append(category_idx)

           return patterns, labels

       async def run_pattern_simulation(self, pattern_type):
           """Run simulation to generate specific pattern type."""
           # Configure simulation parameters based on desired pattern
           if pattern_type == 'random':
               params = {'sorting_strength': 0.1, 'movement_speed': 2.0}
           elif pattern_type == 'clustered':
               params = {'sorting_strength': 1.0, 'movement_speed': 0.5}
           elif pattern_type == 'striped':
               params = {'sorting_strength': 0.7, 'movement_speed': 1.0, 'anisotropy': True}
           elif pattern_type == 'spotted':
               params = {'sorting_strength': 1.2, 'movement_speed': 0.3}
           elif pattern_type == 'spiral':
               params = {'sorting_strength': 0.8, 'rotation_bias': 0.3}

           # Run simulation
           simulation = PatternGenerationSimulation(params)
           result = await simulation.run(timesteps=500)

           return result['final_pattern']

       def train_classifier(self, train_patterns, train_labels, epochs=50):
           """Train the pattern classifier."""
           # Create dataset and dataloader
           dataset = MorphogenesisPatternDataset(train_patterns, train_labels)
           dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

           self.classifier.train()
           training_losses = []

           for epoch in range(epochs):
               epoch_loss = 0.0

               for batch_patterns, batch_labels in dataloader:
                   # Forward pass
                   outputs = self.classifier(batch_patterns)
                   loss = self.criterion(outputs, batch_labels)

                   # Backward pass
                   self.optimizer.zero_grad()
                   loss.backward()
                   self.optimizer.step()

                   epoch_loss += loss.item()

               avg_loss = epoch_loss / len(dataloader)
               training_losses.append(avg_loss)

               if (epoch + 1) % 10 == 0:
                   print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

           return training_losses

       def analyze_pattern(self, pattern):
           """Analyze a pattern and return classification probabilities."""
           self.classifier.eval()

           pattern_tensor = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0)

           with torch.no_grad():
               outputs = self.classifier(pattern_tensor)
               probabilities = F.softmax(outputs, dim=1)

           # Convert to category predictions
           predictions = {}
           for i, category in enumerate(self.pattern_categories):
               predictions[category] = probabilities[0][i].item()

           return predictions

Step 4: Automated Discovery System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a system that automatically discovers new morphogenetic algorithms.

.. code-block:: python

   import itertools
   from sklearn.cluster import KMeans
   from sklearn.metrics import silhouette_score

   class MorphogenesisAutoDiscovery:
       def __init__(self):
           self.discovered_algorithms = []
           self.algorithm_performance = {}

           # Algorithm building blocks
           self.movement_rules = [
               'random_walk', 'gradient_following', 'neighbor_attraction',
               'neighbor_repulsion', 'boundary_following'
           ]

           self.decision_rules = [
               'threshold_based', 'probabilistic', 'learning_based',
               'consensus_based', 'energy_minimizing'
           ]

           self.communication_modes = [
               'none', 'local_signaling', 'global_broadcast',
               'chemical_trails', 'mechanical_forces'
           ]

       async def generate_algorithm_variants(self, max_combinations=100):
           """Generate variants of morphogenetic algorithms."""
           algorithms = []

           # Generate combinations of algorithm components
           combinations = itertools.product(
               self.movement_rules,
               self.decision_rules,
               self.communication_modes
           )

           for i, (movement, decision, communication) in enumerate(combinations):
               if i >= max_combinations:
                   break

               algorithm = MorphogeneticAlgorithmVariant(
                   algorithm_id=f"variant_{i}",
                   movement_rule=movement,
                   decision_rule=decision,
                   communication_mode=communication
               )

               algorithms.append(algorithm)

           return algorithms

       async def evaluate_algorithm_performance(self, algorithm, test_scenarios):
           """Evaluate algorithm performance across multiple scenarios."""
           performance_scores = []

           for scenario in test_scenarios:
               score = await self.run_evaluation_scenario(algorithm, scenario)
               performance_scores.append(score)

           # Aggregate performance metrics
           avg_performance = np.mean(performance_scores)
           performance_stability = 1.0 / (1.0 + np.std(performance_scores))

           return {
               'average_performance': avg_performance,
               'performance_stability': performance_stability,
               'scenario_scores': performance_scores
           }

       async def run_evaluation_scenario(self, algorithm, scenario):
           """Run single evaluation scenario."""
           # Initialize simulation with algorithm
           simulation = MorphogenesisSimulation(
               algorithm=algorithm,
               scenario_config=scenario
           )

           # Run simulation
           results = await simulation.run(max_timesteps=1000)

           # Calculate performance score based on scenario objectives
           score = 0.0

           if scenario['objective'] == 'sorting':
               score = results['final_sorting_quality']
           elif scenario['objective'] == 'pattern_formation':
               score = results['pattern_complexity_score']
           elif scenario['objective'] == 'collective_movement':
               score = results['movement_coordination_score']

           return score

       async def discover_novel_algorithms(self, generations=10):
           """Use evolutionary approach to discover novel algorithms."""
           print("Starting automated algorithm discovery...")

           # Define test scenarios
           test_scenarios = self.create_evaluation_scenarios()

           # Initialize population
           population = await self.generate_algorithm_variants(50)

           discovery_history = []

           for generation in range(generations):
               print(f"Generation {generation + 1}/{generations}")

               # Evaluate all algorithms
               evaluation_results = []
               for algorithm in population:
                   performance = await self.evaluate_algorithm_performance(
                       algorithm, test_scenarios
                   )
                   evaluation_results.append(performance)

                   # Store performance
                   self.algorithm_performance[algorithm.algorithm_id] = performance

               # Select best algorithms
               performance_scores = [r['average_performance'] for r in evaluation_results]
               best_indices = np.argsort(performance_scores)[-20:]  # Keep top 20

               elite_population = [population[i] for i in best_indices]

               # Generate new variants through mutation and crossover
               new_population = elite_population.copy()

               while len(new_population) < 50:
                   if len(elite_population) >= 2:
                       parent1, parent2 = np.random.choice(elite_population, 2, replace=False)
                       child = await self.crossover_algorithms(parent1, parent2)
                       child = await self.mutate_algorithm(child)
                       new_population.append(child)

               population = new_population

               # Record discovery progress
               best_performance = max(performance_scores)
               discovery_history.append({
                   'generation': generation,
                   'best_performance': best_performance,
                   'population_diversity': self.calculate_population_diversity(population)
               })

               print(f"  Best performance: {best_performance:.4f}")

           # Identify most promising discoveries
           final_rankings = sorted(
               self.algorithm_performance.items(),
               key=lambda x: x[1]['average_performance'],
               reverse=True
           )

           return {
               'top_algorithms': final_rankings[:10],
               'discovery_history': discovery_history,
               'total_algorithms_tested': len(self.algorithm_performance)
           }

       async def crossover_algorithms(self, parent1, parent2):
           """Create child algorithm by combining parent algorithms."""
           # Random combination of parent traits
           child_movement = np.random.choice([parent1.movement_rule, parent2.movement_rule])
           child_decision = np.random.choice([parent1.decision_rule, parent2.decision_rule])
           child_communication = np.random.choice([parent1.communication_mode, parent2.communication_mode])

           child_id = f"crossover_{len(self.discovered_algorithms)}"

           child = MorphogeneticAlgorithmVariant(
               algorithm_id=child_id,
               movement_rule=child_movement,
               decision_rule=child_decision,
               communication_mode=child_communication
           )

           return child

       async def mutate_algorithm(self, algorithm):
           """Mutate algorithm by changing one component randomly."""
           mutation_rate = 0.3

           mutated = algorithm.copy()

           if np.random.random() < mutation_rate:
               mutated.movement_rule = np.random.choice(self.movement_rules)

           if np.random.random() < mutation_rate:
               mutated.decision_rule = np.random.choice(self.decision_rules)

           if np.random.random() < mutation_rate:
               mutated.communication_mode = np.random.choice(self.communication_modes)

           mutated.algorithm_id = f"mutant_{len(self.discovered_algorithms)}"

           return mutated

Step 5: Running ML-Enhanced Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Put everything together in a complete ML-enhanced morphogenesis system.

.. code-block:: python

   async def run_ml_enhanced_morphogenesis():
       """Complete ML-enhanced morphogenesis simulation."""
       print("Starting ML-enhanced morphogenesis simulation...")

       # Phase 1: Train pattern classifier
       print("\nPhase 1: Training pattern recognition system...")
       pattern_system = PatternAnalysisSystem()

       # Generate training data
       train_patterns, train_labels = await pattern_system.generate_training_data(500)

       # Train classifier
       training_losses = pattern_system.train_classifier(train_patterns, train_labels, epochs=30)

       # Phase 2: RL-based strategy discovery
       print("\nPhase 2: Discovering optimal strategies with RL...")
       rl_env = MorphogenesisRLEnvironment(grid_size=(40, 40), num_agents=50)

       # Train RL agents
       num_episodes = 100
       episode_rewards = []

       for episode in range(num_episodes):
           state = await rl_env.reset()
           episode_reward = 0

           for step in range(200):
               # Get actions from RL agents
               actions = []
               for agent in rl_env.agents:
                   perception = await agent.perceive_environment()
                   action = await agent.choose_rl_action(perception.numpy())
                   actions.append(action)

               # Execute environment step
               next_state, rewards, done, info = await rl_env.step(actions)

               # Store experiences and learn
               for agent, reward in zip(rl_env.agents, rewards):
                   # Store experience for learning
                   pass  # Implementation would store and learn from experience

               episode_reward += np.mean(rewards)

               if done:
                   break

           episode_rewards.append(episode_reward)

           if (episode + 1) % 20 == 0:
               avg_reward = np.mean(episode_rewards[-20:])
               print(f"Episode {episode + 1}: Average reward = {avg_reward:.3f}")

       # Phase 3: Automated algorithm discovery
       print("\nPhase 3: Discovering novel algorithms...")
       discovery_system = MorphogenesisAutoDiscovery()

       discovery_results = await discovery_system.discover_novel_algorithms(generations=5)

       print(f"\nDiscovery Results:")
       print(f"Total algorithms tested: {discovery_results['total_algorithms_tested']}")
       print("\nTop 5 discovered algorithms:")

       for i, (alg_id, performance) in enumerate(discovery_results['top_algorithms'][:5]):
           print(f"{i+1}. {alg_id}: {performance['average_performance']:.4f}")

       # Phase 4: Integration and final simulation
       print("\nPhase 4: Running integrated ML-enhanced simulation...")

       # Use best discovered algorithm with ML-controlled agents
       best_algorithm = discovery_results['top_algorithms'][0][0]

       # Create hybrid system
       hybrid_simulation = MLMorphogenesisSimulation(
           pattern_classifier=pattern_system.classifier,
           rl_agents=rl_env.agents[:10],  # Use trained RL agents
           discovered_algorithm=best_algorithm
       )

       # Run final simulation
       final_results = await hybrid_simulation.run(timesteps=500)

       return {
           'pattern_training_losses': training_losses,
           'rl_episode_rewards': episode_rewards,
           'discovery_results': discovery_results,
           'final_simulation_results': final_results
       }

   # Run the complete ML-enhanced system
   if __name__ == "__main__":
       results = asyncio.run(run_ml_enhanced_morphogenesis())

Tutorial 2: Distributed Computing for Large-Scale Simulations
--------------------------------------------------------------

Learn how to scale morphogenesis simulations to massive scales using distributed computing techniques.

**Learning Objectives:**
- Implement distributed agent coordination
- Use message passing for inter-process communication
- Scale simulations across multiple machines
- Handle load balancing and fault tolerance

**Prerequisites:**
- Understanding of parallel computing concepts
- Familiarity with distributed systems
- Knowledge of MPI or similar frameworks

Step 1: Distributed Architecture Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design a distributed architecture that can scale to millions of agents.

.. code-block:: python

   import asyncio
   import multiprocessing as mp
   import pickle
   import zmq
   from mpi4py import MPI
   import numpy as np

   class DistributedMorphogenesisSystem:
       def __init__(self, total_agents, num_processes):
           self.total_agents = total_agents
           self.num_processes = num_processes
           self.agents_per_process = total_agents // num_processes

           # MPI setup
           self.comm = MPI.COMM_WORLD
           self.rank = self.comm.Get_rank()
           self.size = self.comm.Get_size()

           # ZeroMQ for high-performance messaging
           self.context = zmq.Context()
           self.message_socket = None

           # Spatial partitioning
           self.domain_partition = self.calculate_domain_partition()

           # Local agents managed by this process
           self.local_agents = []
           self.neighbor_agents = {}  # Agents from neighboring processes

       def calculate_domain_partition(self):
           """Calculate spatial domain partition for this process."""
           # Assuming 2D grid partitioning
           grid_dims = int(np.sqrt(self.size))

           process_x = self.rank % grid_dims
           process_y = self.rank // grid_dims

           # Calculate local domain boundaries
           domain_width = 100 // grid_dims  # Assuming 100x100 global domain
           domain_height = 100 // grid_dims

           return {
               'x_min': process_x * domain_width,
               'x_max': (process_x + 1) * domain_width,
               'y_min': process_y * domain_height,
               'y_max': (process_y + 1) * domain_height,
               'neighbors': self.get_neighboring_processes(process_x, process_y, grid_dims)
           }

       def get_neighboring_processes(self, px, py, grid_dims):
           """Get neighboring process ranks."""
           neighbors = []

           for dx in [-1, 0, 1]:
               for dy in [-1, 0, 1]:
                   if dx == 0 and dy == 0:
                       continue

                   nx, ny = px + dx, py + dy

                   if 0 <= nx < grid_dims and 0 <= ny < grid_dims:
                       neighbor_rank = ny * grid_dims + nx
                       neighbors.append(neighbor_rank)

           return neighbors

       async def initialize_distributed_system(self):
           """Initialize distributed components."""
           # Setup ZeroMQ communication
           self.setup_communication()

           # Create local agents
           await self.create_local_agents()

           # Initialize neighboring agent tracking
           self.neighbor_agents = {rank: [] for rank in self.domain_partition['neighbors']}

       def setup_communication(self):
           """Setup ZeroMQ communication channels."""
           # Publisher for broadcasting agent updates
           self.publisher = self.context.socket(zmq.PUB)
           pub_port = 5555 + self.rank
           self.publisher.bind(f"tcp://*:{pub_port}")

           # Subscriber for receiving neighbor updates
           self.subscriber = self.context.socket(zmq.SUB)

           for neighbor_rank in self.domain_partition['neighbors']:
               neighbor_port = 5555 + neighbor_rank
               self.subscriber.connect(f"tcp://localhost:{neighbor_port}")

               # Subscribe to updates from this neighbor
               topic = f"agents_{neighbor_rank}".encode('utf-8')
               self.subscriber.setsockopt(zmq.SUBSCRIBE, topic)

       async def create_local_agents(self):
           """Create agents assigned to this process."""
           for i in range(self.agents_per_process):
               agent_id = self.rank * self.agents_per_process + i

               # Random position within local domain
               position = (
                   np.random.uniform(self.domain_partition['x_min'],
                                   self.domain_partition['x_max']),
                   np.random.uniform(self.domain_partition['y_min'],
                                   self.domain_partition['y_max'])
               )

               agent = DistributedMorphogenesisAgent(
                   agent_id=agent_id,
                   initial_position=position,
                   parent_system=self
               )

               self.local_agents.append(agent)

       async def distributed_simulation_step(self):
           """Execute one distributed simulation step."""
           # Phase 1: Update local agents
           for agent in self.local_agents:
               await agent.step()

           # Phase 2: Exchange boundary information
           await self.exchange_boundary_agents()

           # Phase 3: Process cross-boundary interactions
           await self.process_cross_boundary_interactions()

           # Phase 4: Synchronization barrier
           self.comm.Barrier()

       async def exchange_boundary_agents(self):
           """Exchange information about agents near domain boundaries."""
           # Find agents near boundaries
           boundary_agents = []
           boundary_threshold = 5.0  # Distance from boundary

           for agent in self.local_agents:
               x, y = agent.position

               # Check if near any boundary
               near_boundary = (
                   x - self.domain_partition['x_min'] < boundary_threshold or
                   self.domain_partition['x_max'] - x < boundary_threshold or
                   y - self.domain_partition['y_min'] < boundary_threshold or
                   self.domain_partition['y_max'] - y < boundary_threshold
               )

               if near_boundary:
                   boundary_agents.append(agent.serialize_for_communication())

           # Publish boundary agent information
           message_topic = f"agents_{self.rank}".encode('utf-8')
           message_data = pickle.dumps(boundary_agents)

           self.publisher.send_multipart([message_topic, message_data])

           # Receive updates from neighbors
           await self.receive_neighbor_updates()

       async def receive_neighbor_updates(self):
           """Receive and process updates from neighboring processes."""
           try:
               # Non-blocking receive
               topic, message_data = self.subscriber.recv_multipart(zmq.NOBLOCK)

               # Deserialize neighbor agents
               neighbor_agents = pickle.loads(message_data)

               # Extract source rank from topic
               source_rank = int(topic.decode('utf-8').split('_')[1])

               # Update neighbor agent information
               self.neighbor_agents[source_rank] = neighbor_agents

           except zmq.Again:
               # No messages available, continue
               pass

       async def process_cross_boundary_interactions(self):
           """Process interactions between local and neighbor agents."""
           for agent in self.local_agents:
               # Check interactions with neighbor agents
               for neighbor_rank, neighbor_agents in self.neighbor_agents.items():
                   for neighbor_data in neighbor_agents:
                       neighbor_pos = neighbor_data['position']
                       distance = np.linalg.norm(
                           np.array(agent.position) - np.array(neighbor_pos)
                       )

                       if distance < agent.interaction_radius:
                           await agent.process_neighbor_interaction(neighbor_data)

   class DistributedMorphogenesisAgent:
       def __init__(self, agent_id, initial_position, parent_system):
           self.agent_id = agent_id
           self.position = initial_position
           self.parent_system = parent_system
           self.interaction_radius = 3.0

           # Agent-specific properties
           self.cell_type = f"type_{agent_id % 3}"
           self.energy = 1.0
           self.age = 0

       async def step(self):
           """Execute agent step in distributed environment."""
           # Standard agent behavior
           await self.update_position()
           await self.update_internal_state()

           # Handle domain boundary crossing
           await self.check_domain_boundaries()

       async def update_position(self):
           """Update agent position based on local rules."""
           # Simple random walk with bias toward other agents
           movement = np.random.normal(0, 0.5, 2)

           # Apply movement
           new_position = (
               self.position[0] + movement[0],
               self.position[1] + movement[1]
           )

           self.position = new_position

       async def check_domain_boundaries(self):
           """Check if agent has crossed domain boundaries."""
           x, y = self.position
           domain = self.parent_system.domain_partition

           # Check if agent has left local domain
           if (x < domain['x_min'] or x >= domain['x_max'] or
               y < domain['y_min'] or y >= domain['y_max']):

               # Agent needs to be migrated to another process
               await self.initiate_migration()

       async def initiate_migration(self):
           """Initiate migration to appropriate process."""
           # Determine destination process
           target_rank = self.calculate_target_process()

           if target_rank is not None and target_rank != self.parent_system.rank:
               # Serialize agent for migration
               agent_data = self.serialize_for_migration()

               # Send migration request
               migration_message = {
                   'type': 'agent_migration',
                   'agent_data': agent_data,
                   'source_rank': self.parent_system.rank
               }

               # Use MPI for reliable migration
               self.parent_system.comm.isend(migration_message, dest=target_rank, tag=1)

               # Mark agent for removal from local system
               self.parent_system.local_agents.remove(self)

       def serialize_for_communication(self):
           """Serialize agent data for inter-process communication."""
           return {
               'agent_id': self.agent_id,
               'position': self.position,
               'cell_type': self.cell_type,
               'energy': self.energy,
               'age': self.age
           }

       def serialize_for_migration(self):
           """Serialize complete agent state for migration."""
           return {
               'agent_id': self.agent_id,
               'position': self.position,
               'cell_type': self.cell_type,
               'energy': self.energy,
               'age': self.age,
               'interaction_radius': self.interaction_radius,
               # Include any other necessary state
           }

**Exercise:** Implement load balancing to redistribute agents when load becomes uneven.

Step 2: Fault Tolerance and Recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement fault tolerance mechanisms for robust distributed simulations.

.. code-block:: python

   class FaultTolerantDistributedSystem(DistributedMorphogenesisSystem):
       def __init__(self, total_agents, num_processes, checkpoint_interval=100):
           super().__init__(total_agents, num_processes)
           self.checkpoint_interval = checkpoint_interval
           self.current_timestep = 0
           self.checkpoints = {}
           self.failed_processes = set()

       async def run_fault_tolerant_simulation(self, max_timesteps=1000):
           """Run simulation with fault tolerance."""
           print(f"Process {self.rank}: Starting fault-tolerant simulation...")

           await self.initialize_distributed_system()

           for timestep in range(max_timesteps):
               self.current_timestep = timestep

               try:
                   # Execute simulation step
                   await self.distributed_simulation_step()

                   # Periodic checkpointing
                   if timestep % self.checkpoint_interval == 0:
                       await self.create_checkpoint(timestep)

                   # Check for failed processes
                   await self.monitor_process_health()

               except Exception as e:
                   print(f"Process {self.rank}: Error at timestep {timestep}: {e}")

                   # Attempt recovery
                   await self.handle_process_failure()

       async def create_checkpoint(self, timestep):
           """Create checkpoint of current simulation state."""
           checkpoint_data = {
               'timestep': timestep,
               'local_agents': [agent.serialize_for_migration() for agent in self.local_agents],
               'domain_partition': self.domain_partition,
               'random_state': np.random.get_state()
           }

           # Store checkpoint locally
           checkpoint_filename = f"checkpoint_{self.rank}_{timestep}.pkl"
           with open(checkpoint_filename, 'wb') as f:
               pickle.dump(checkpoint_data, f)

           # Also send to redundant storage process
           if self.size > 4:  # Only if we have enough processes
               backup_rank = (self.rank + self.size // 2) % self.size

               backup_message = {
                   'type': 'checkpoint_backup',
                   'checkpoint_data': checkpoint_data,
                   'source_rank': self.rank
               }

               self.comm.isend(backup_message, dest=backup_rank, tag=2)

       async def monitor_process_health(self):
           """Monitor health of other processes."""
           # Send heartbeat to all processes
           heartbeat = {'rank': self.rank, 'timestep': self.current_timestep}

           for other_rank in range(self.size):
               if other_rank != self.rank:
                   try:
                       self.comm.isend(heartbeat, dest=other_rank, tag=3)
                   except:
                       # Process might be failed
                       self.failed_processes.add(other_rank)

       async def handle_process_failure(self):
           """Handle failure of a process."""
           if len(self.failed_processes) > 0:
               print(f"Process {self.rank}: Detected {len(self.failed_processes)} failed processes")

               # Redistribute work from failed processes
               await self.redistribute_failed_work()

               # Update domain partitions
               self.update_domain_partitions_for_failures()

       async def redistribute_failed_work(self):
           """Redistribute work from failed processes."""
           surviving_processes = [r for r in range(self.size) if r not in self.failed_processes]

           # Each surviving process takes responsibility for some failed work
           for failed_rank in self.failed_processes:
               if self.rank == min(surviving_processes):  # Coordinator process
                   # Load checkpoint for failed process
                   checkpoint_data = await self.load_checkpoint_for_process(failed_rank)

                   if checkpoint_data:
                       # Distribute failed agents among surviving processes
                       failed_agents = checkpoint_data['local_agents']
                       agents_per_survivor = len(failed_agents) // len(surviving_processes)

                       for i, survivor_rank in enumerate(surviving_processes):
                           start_idx = i * agents_per_survivor
                           end_idx = start_idx + agents_per_survivor

                           if i == len(surviving_processes) - 1:  # Last process gets remainder
                               end_idx = len(failed_agents)

                           assigned_agents = failed_agents[start_idx:end_idx]

                           if survivor_rank == self.rank:
                               # Add agents to local system
                               for agent_data in assigned_agents:
                                   agent = self.reconstruct_agent_from_data(agent_data)
                                   self.local_agents.append(agent)

Step 3: Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize the distributed system for maximum performance.

.. code-block:: python

   class OptimizedDistributedSystem(FaultTolerantDistributedSystem):
       def __init__(self, total_agents, num_processes):
           super().__init__(total_agents, num_processes)

           # Performance monitoring
           self.performance_metrics = {
               'communication_time': 0.0,
               'computation_time': 0.0,
               'synchronization_time': 0.0,
               'load_balance_score': 1.0
           }

           # Adaptive load balancing
           self.load_balancer = AdaptiveLoadBalancer(self)

       async def optimized_simulation_step(self):
           """Optimized version of simulation step with performance monitoring."""
           import time

           # Computation phase
           comp_start = time.time()
           for agent in self.local_agents:
               await agent.step()
           comp_time = time.time() - comp_start

           # Communication phase
           comm_start = time.time()
           await self.optimized_boundary_exchange()
           comm_time = time.time() - comm_start

           # Synchronization phase
           sync_start = time.time()
           self.comm.Barrier()
           sync_time = time.time() - sync_start

           # Update performance metrics
           self.performance_metrics['computation_time'] += comp_time
           self.performance_metrics['communication_time'] += comm_time
           self.performance_metrics['synchronization_time'] += sync_time

           # Periodic load balancing
           if self.current_timestep % 50 == 0:
               await self.load_balancer.balance_load()

       async def optimized_boundary_exchange(self):
           """Optimized boundary information exchange."""
           # Use non-blocking MPI operations for better overlap
           send_requests = []
           recv_requests = []

           # Prepare boundary data
           boundary_data = self.prepare_boundary_data()

           # Initiate non-blocking sends
           for neighbor_rank in self.domain_partition['neighbors']:
               req = self.comm.isend(boundary_data, dest=neighbor_rank, tag=4)
               send_requests.append(req)

           # Initiate non-blocking receives
           neighbor_data = {}
           for neighbor_rank in self.domain_partition['neighbors']:
               req = self.comm.irecv(source=neighbor_rank, tag=4)
               recv_requests.append((neighbor_rank, req))

           # Process received data as it arrives
           for neighbor_rank, req in recv_requests:
               data = req.wait()
               neighbor_data[neighbor_rank] = data

           # Wait for all sends to complete
           MPI.Request.waitall(send_requests)

           # Update neighbor information
           for neighbor_rank, data in neighbor_data.items():
               self.neighbor_agents[neighbor_rank] = data

   class AdaptiveLoadBalancer:
       def __init__(self, distributed_system):
           self.system = distributed_system
           self.load_history = []

       async def balance_load(self):
           """Perform adaptive load balancing."""
           # Gather load information from all processes
           local_load = len(self.system.local_agents)
           all_loads = self.system.comm.allgather(local_load)

           # Calculate load imbalance
           avg_load = sum(all_loads) / len(all_loads)
           max_load = max(all_loads)
           min_load = min(all_loads)

           load_balance_score = min_load / max(max_load, 1)
           self.system.performance_metrics['load_balance_score'] = load_balance_score

           # If load is significantly imbalanced, redistribute
           if load_balance_score < 0.8:  # Threshold for load balancing
               await self.redistribute_agents(all_loads)

       async def redistribute_agents(self, all_loads):
           """Redistribute agents to balance load."""
           avg_load = sum(all_loads) / len(all_loads)

           # Processes with excess agents
           if len(self.system.local_agents) > avg_load * 1.2:
               excess_agents = int(len(self.system.local_agents) - avg_load)

               # Find processes that need agents
               for rank, load in enumerate(all_loads):
                   if load < avg_load * 0.8 and rank != self.system.rank:
                       # Send excess agents to this process
                       agents_to_send = min(excess_agents, int(avg_load - load))

                       for _ in range(agents_to_send):
                           if self.system.local_agents:
                               agent = self.system.local_agents.pop()
                               agent_data = agent.serialize_for_migration()

                               migration_message = {
                                   'type': 'load_balance_migration',
                                   'agent_data': agent_data
                               }

                               self.system.comm.isend(migration_message, dest=rank, tag=5)

                       excess_agents -= agents_to_send
                       if excess_agents <= 0:
                           break

**Exercise:** Implement dynamic domain repartitioning based on agent density.

Summary
-------

These advanced tutorials have covered:

1. **Machine Learning Integration:**
   - Neural network-controlled cellular agents
   - Reinforcement learning for strategy optimization
   - Deep learning for pattern recognition
   - Automated algorithm discovery

2. **Distributed Computing:**
   - Large-scale distributed architectures
   - Message passing and synchronization
   - Fault tolerance and recovery mechanisms
   - Performance optimization techniques

**Key Achievements:**
- Learned to integrate cutting-edge AI techniques with morphogenesis
- Developed skills for massive-scale simulations
- Understood advanced optimization and discovery methods
- Gained experience with fault-tolerant distributed systems

**Next Steps:**
- Explore research-level applications in your domain
- Contribute to open-source morphogenesis platforms
- Develop novel algorithms using the learned techniques
- Apply these methods to real-world biological problems

These advanced techniques represent the current frontier of computational morphogenesis research and provide the foundation for breakthrough discoveries in understanding biological development and creating bio-inspired technologies.