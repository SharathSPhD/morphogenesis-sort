"""Adaptive learning cell agent implementations.

This module provides cell agents that can learn and adapt their behavior
based on experience, including reinforcement learning, imitation learning,
and evolutionary optimization strategies.
"""

import asyncio
import math
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from ..cell_agent import AsyncCellAgent
from ...data.types import CellID, CellData, CellParameters, Position
from ...data.actions import CellAction, create_wait_action, create_move_action, create_swap_action
from ...data.state import SimulationState
from .config import AdaptiveConfig, LearningStrategy


@dataclass
class Experience:
    """A learning experience (state, action, reward, next_state)."""
    state: Dict[str, float]
    action: str
    reward: float
    next_state: Dict[str, float]
    timestamp: int
    success: bool = False


@dataclass
class PolicyParameters:
    """Parameters for the agent's policy."""
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    bias: float = 0.0
    exploration_rate: float = 0.1
    confidence: float = 0.5


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    total_reward: float = 0.0
    average_reward: float = 0.0
    success_rate: float = 0.0
    learning_progress: float = 0.0
    adaptation_events: int = 0


class ActionType(Enum):
    """Types of actions the adaptive agent can take."""
    WAIT = "wait"
    MOVE = "move"
    SWAP = "swap"
    EXPLORE = "explore"
    IMITATE = "imitate"


class AdaptiveCellAgent(AsyncCellAgent):
    """Cell agent with adaptive learning capabilities.

    This agent can learn from experience using various strategies including
    reinforcement learning, imitation learning, and evolutionary optimization.
    It adapts its behavior based on rewards and observations of other agents.
    """

    def __init__(
        self,
        cell_id: CellID,
        initial_data: CellData,
        adaptive_config: AdaptiveConfig,
        parameters: CellParameters,
        random_seed: Optional[int] = None
    ):
        # Convert AdaptiveConfig to CellBehaviorConfig for parent
        from ..cell_agent import CellBehaviorConfig
        behavior_config = CellBehaviorConfig(
            behavior_type="adaptive",
            decision_frequency=adaptive_config.decision_frequency,
            action_delay=adaptive_config.action_delay,
            movement_enabled=adaptive_config.movement_enabled,
            max_movement_distance=adaptive_config.max_movement_distance,
            movement_randomness=adaptive_config.movement_randomness,
            interaction_enabled=adaptive_config.interaction_enabled,
            cooperation_probability=adaptive_config.cooperation_probability,
            learning_enabled=adaptive_config.learning_enabled,
            learning_rate=adaptive_config.learning_rate,
            memory_capacity=adaptive_config.memory_capacity,
            error_recovery=adaptive_config.error_recovery,
            max_retry_attempts=adaptive_config.max_retry_attempts,
            freeze_on_error=adaptive_config.freeze_on_error,
        )

        super().__init__(cell_id, initial_data, behavior_config, parameters, random_seed)

        self.adaptive_config = adaptive_config
        self.performance_metrics = PerformanceMetrics()

        # Learning components
        self.experience_buffer: deque = deque(maxlen=adaptive_config.memory_capacity)
        self.policy_params = PolicyParameters()
        self.reward_history: deque = deque(maxlen=adaptive_config.performance_window)

        # State representation
        self.state_features: List[str] = [
            'position_x', 'position_y', 'sort_value', 'age',
            'neighbor_count', 'local_density', 'local_disorder',
            'recent_reward', 'exploration_bonus'
        ]

        # Action space
        self.action_space = list(ActionType)
        self.action_values: Dict[str, float] = {action.value: 0.0 for action in ActionType}

        # Imitation learning
        self.role_models: Dict[CellID, float] = {}  # CellID -> performance score
        self.imitation_buffer: deque = deque(maxlen=50)

        # Evolutionary components (for evolution strategy)
        self.population_members: List[PolicyParameters] = []
        self.fitness_scores: List[float] = []

        # Initialize learning system
        self._initialize_learning_system()

    async def _setup_behavior(self) -> None:
        """Setup adaptive learning behavior."""
        await super()._setup_behavior()

        # Initialize policy parameters
        await self._initialize_policy()

        # Setup reward system
        await self._setup_reward_system()

        # Initialize neural network if configured
        if self.adaptive_config.use_neural_network:
            await self._initialize_neural_network()

    def _initialize_learning_system(self) -> None:
        """Initialize the learning system components."""
        # Initialize policy parameters
        feature_count = len(self.state_features)
        action_count = len(self.action_space)

        if self.adaptive_config.learning_strategy == LearningStrategy.REINFORCEMENT:
            # Q-learning: action_count values for each state
            self.policy_params.weights = np.random.normal(0, 0.1, (feature_count, action_count))
        else:
            # Policy gradient: single policy
            self.policy_params.weights = np.random.normal(0, 0.1, feature_count)

        self.policy_params.exploration_rate = self.adaptive_config.exploration_rate

    async def _initialize_policy(self) -> None:
        """Initialize the agent's policy."""
        # Set initial exploration rate
        self.policy_params.exploration_rate = self.adaptive_config.exploration_rate

    async def _setup_reward_system(self) -> None:
        """Setup the reward calculation system."""
        # Initialize reward function weights
        num_functions = len(self.adaptive_config.reward_functions)
        num_weights = len(self.adaptive_config.reward_weights)

        if num_weights < num_functions:
            # Pad with equal weights
            additional_weights = [1.0] * (num_functions - num_weights)
            self.adaptive_config.reward_weights.extend(additional_weights)

    async def _initialize_neural_network(self) -> None:
        """Initialize neural network if using neural network approach."""
        # This would initialize a simple neural network
        # For now, using linear approximation
        pass

    async def _decide_action(self) -> CellAction:
        """Decide action using learned policy."""
        if not self.adaptive_config.learning_enabled:
            return create_wait_action(self.cell_id, self.last_update_timestep)

        # Get current state representation
        current_state = await self._get_state_representation()

        # Choose action based on learning strategy
        action_type = await self._choose_action(current_state)

        # Execute the chosen action
        action = await self._execute_chosen_action(action_type)

        # Store experience for learning (state will be updated after action)
        if len(self.experience_buffer) > 0:
            # Calculate reward for previous action
            previous_experience = self.experience_buffer[-1]
            reward = await self._calculate_reward(previous_experience.state, current_state)

            # Update previous experience with reward and next state
            previous_experience.reward = reward
            previous_experience.next_state = current_state

            # Learn from the experience
            if self.adaptive_config.learning_enabled:
                await self._learn_from_experience(previous_experience)

        # Create new experience (reward will be filled in next decision)
        new_experience = Experience(
            state=current_state,
            action=action_type.value,
            reward=0.0,  # Will be calculated next time
            next_state={},
            timestamp=self.last_update_timestep
        )
        self.experience_buffer.append(new_experience)

        return action

    async def _get_state_representation(self) -> Dict[str, float]:
        """Get current state as feature vector."""
        neighbors = self.get_neighbors()

        state = {
            'position_x': self.current_data.position.x,
            'position_y': self.current_data.position.y,
            'sort_value': self.current_data.sort_value,
            'age': float(self.current_data.age),
            'neighbor_count': float(len(neighbors)),
            'local_density': self.calculate_local_density(),
            'local_disorder': self._calculate_local_disorder(neighbors),
            'recent_reward': self._get_recent_average_reward(),
            'exploration_bonus': self._calculate_exploration_bonus(),
        }

        return state

    async def _choose_action(self, state: Dict[str, float]) -> ActionType:
        """Choose action based on current policy and exploration strategy."""
        if self.adaptive_config.exploration_strategy == "epsilon_greedy":
            return await self._epsilon_greedy_action(state)
        elif self.adaptive_config.exploration_strategy == "ucb":
            return await self._ucb_action(state)
        elif self.adaptive_config.exploration_strategy == "thompson":
            return await self._thompson_sampling_action(state)
        else:
            return await self._epsilon_greedy_action(state)

    async def _epsilon_greedy_action(self, state: Dict[str, float]) -> ActionType:
        """Choose action using epsilon-greedy exploration."""
        if self.rng.random() < self.policy_params.exploration_rate:
            # Explore: random action
            return self.rng.choice(self.action_space)
        else:
            # Exploit: best known action
            return await self._get_best_action(state)

    async def _ucb_action(self, state: Dict[str, float]) -> ActionType:
        """Choose action using Upper Confidence Bound."""
        # This is a simplified UCB implementation
        best_action = None
        best_value = float('-inf')

        for action_type in self.action_space:
            q_value = await self._get_q_value(state, action_type)

            # Add confidence interval
            action_count = self.action_values.get(action_type.value + "_count", 1)
            total_actions = sum(self.action_values.get(a.value + "_count", 1) for a in self.action_space)

            confidence = math.sqrt(2 * math.log(total_actions) / action_count)
            ucb_value = q_value + confidence

            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action_type

        return best_action or ActionType.WAIT

    async def _thompson_sampling_action(self, state: Dict[str, float]) -> ActionType:
        """Choose action using Thompson sampling."""
        # Sample from posterior distributions of action values
        sampled_values = {}

        for action_type in self.action_space:
            mean_value = await self._get_q_value(state, action_type)
            # Assume normal distribution with decreasing variance over time
            variance = max(0.1, 1.0 / (self.decision_counter + 1))
            sampled_value = self.rng.gauss(mean_value, math.sqrt(variance))
            sampled_values[action_type] = sampled_value

        # Return action with highest sampled value
        best_action = max(sampled_values.keys(), key=lambda a: sampled_values[a])
        return best_action

    async def _get_best_action(self, state: Dict[str, float]) -> ActionType:
        """Get the best action according to current policy."""
        best_action = None
        best_value = float('-inf')

        for action_type in self.action_space:
            q_value = await self._get_q_value(state, action_type)
            if q_value > best_value:
                best_value = q_value
                best_action = action_type

        return best_action or ActionType.WAIT

    async def _get_q_value(self, state: Dict[str, float], action: ActionType) -> float:
        """Get Q-value for state-action pair."""
        # Convert state to feature vector
        features = np.array([state[feature] for feature in self.state_features])

        if self.adaptive_config.learning_strategy == LearningStrategy.REINFORCEMENT:
            # Q-learning: features @ weights[:, action_index]
            action_index = self.action_space.index(action)
            if len(self.policy_params.weights.shape) == 2:
                q_value = np.dot(features, self.policy_params.weights[:, action_index])
            else:
                # Fallback to stored action values
                q_value = self.action_values.get(action.value, 0.0)
        else:
            # Policy gradient: compute action probability
            logits = np.dot(features, self.policy_params.weights) + self.policy_params.bias
            q_value = logits  # Use as value estimate

        return float(q_value)

    async def _execute_chosen_action(self, action_type: ActionType) -> CellAction:
        """Execute the chosen action."""
        if action_type == ActionType.WAIT:
            return create_wait_action(self.cell_id, self.last_update_timestep)

        elif action_type == ActionType.MOVE:
            return await self._execute_move_action()

        elif action_type == ActionType.SWAP:
            return await self._execute_swap_action()

        elif action_type == ActionType.EXPLORE:
            return await self._execute_exploration_action()

        elif action_type == ActionType.IMITATE:
            return await self._execute_imitation_action()

        else:
            return create_wait_action(self.cell_id, self.last_update_timestep)

    async def _execute_move_action(self) -> CellAction:
        """Execute a movement action."""
        # Choose random direction with some bias toward beneficial areas
        angle = self.rng.uniform(0, 2 * math.pi)
        distance = self.rng.uniform(0.5, self.behavior_config.max_movement_distance)

        # Add some intelligence to movement
        neighbors = self.get_neighbors()
        if neighbors:
            # Move toward less crowded area
            crowd_center_x = sum(n.position.x for n in neighbors.values()) / len(neighbors)
            crowd_center_y = sum(n.position.y for n in neighbors.values()) / len(neighbors)

            # Move away from crowd center
            away_x = self.current_data.position.x - crowd_center_x
            away_y = self.current_data.position.y - crowd_center_y
            away_magnitude = math.sqrt(away_x**2 + away_y**2)

            if away_magnitude > 0:
                away_x /= away_magnitude
                away_y /= away_magnitude

                # Blend random with anti-crowd movement
                move_x = 0.7 * distance * math.cos(angle) + 0.3 * distance * away_x
                move_y = 0.7 * distance * math.sin(angle) + 0.3 * distance * away_y
            else:
                move_x = distance * math.cos(angle)
                move_y = distance * math.sin(angle)
        else:
            move_x = distance * math.cos(angle)
            move_y = distance * math.sin(angle)

        target_position = Position(
            self.current_data.position.x + move_x,
            self.current_data.position.y + move_y
        )

        return create_move_action(self.cell_id, self.last_update_timestep, target_position)

    async def _execute_swap_action(self) -> CellAction:
        """Execute a swap action with intelligent neighbor selection."""
        neighbors = self.get_neighbors()
        if not neighbors:
            return create_wait_action(self.cell_id, self.last_update_timestep)

        # Choose neighbor based on learning
        best_neighbor = None
        best_benefit = 0.0

        for neighbor_id, neighbor_data in neighbors.items():
            # Estimate benefit of swapping
            benefit = self._estimate_swap_benefit(neighbor_data)
            if benefit > best_benefit:
                best_benefit = benefit
                best_neighbor = neighbor_id

        if best_neighbor and best_benefit > 0.1:
            return create_swap_action(
                self.cell_id,
                self.last_update_timestep,
                best_neighbor,
                probability=min(1.0, best_benefit)
            )

        return create_wait_action(self.cell_id, self.last_update_timestep)

    async def _execute_exploration_action(self) -> CellAction:
        """Execute an exploration action to gather information."""
        # Exploration: try actions with high uncertainty
        neighbors = self.get_neighbors()

        if neighbors and self.rng.random() < 0.5:
            # Explore by swapping with random neighbor
            neighbor_id = self.rng.choice(list(neighbors.keys()))
            return create_swap_action(
                self.cell_id,
                self.last_update_timestep,
                neighbor_id,
                probability=0.3
            )
        else:
            # Explore by moving to new area
            return await self._execute_move_action()

    async def _execute_imitation_action(self) -> CellAction:
        """Execute action by imitating a successful neighbor."""
        if not self.adaptive_config.imitation_enabled:
            return create_wait_action(self.cell_id, self.last_update_timestep)

        # Find role models
        await self._update_role_models()

        if not self.role_models:
            return create_wait_action(self.cell_id, self.last_update_timestep)

        # Choose best role model
        best_model = max(self.role_models.keys(), key=lambda k: self.role_models[k])

        # Try to imitate their recent action (if observable)
        imitation_action = await self._imitate_model(best_model)
        if imitation_action:
            return imitation_action

        return create_wait_action(self.cell_id, self.last_update_timestep)

    async def _calculate_reward(self, previous_state: Dict[str, float], current_state: Dict[str, float]) -> float:
        """Calculate reward based on state transition."""
        total_reward = 0.0

        for i, reward_function in enumerate(self.adaptive_config.reward_functions):
            weight = self.adaptive_config.reward_weights[i]

            if reward_function == "sorting_progress":
                # Reward for improving local sorting
                prev_disorder = previous_state.get('local_disorder', 0.5)
                curr_disorder = current_state.get('local_disorder', 0.5)
                sorting_reward = (prev_disorder - curr_disorder) * 10.0
                total_reward += weight * sorting_reward

            elif reward_function == "efficiency":
                # Reward for efficient actions (low energy cost)
                position_change = abs(current_state.get('position_x', 0) - previous_state.get('position_x', 0))
                position_change += abs(current_state.get('position_y', 0) - previous_state.get('position_y', 0))
                efficiency_reward = max(0, 1.0 - position_change * 0.1)
                total_reward += weight * efficiency_reward

            elif reward_function == "cooperation":
                # Reward for maintaining good neighbor relationships
                neighbor_change = current_state.get('neighbor_count', 0) - previous_state.get('neighbor_count', 0)
                cooperation_reward = max(0, neighbor_change * 0.5)
                total_reward += weight * cooperation_reward

        # Add intrinsic motivation (curiosity bonus)
        if self.adaptive_config.intrinsic_motivation:
            curiosity_reward = self._calculate_exploration_bonus() * self.adaptive_config.curiosity_bonus
            total_reward += curiosity_reward

        return total_reward

    async def _learn_from_experience(self, experience: Experience) -> None:
        """Learn from a single experience."""
        if self.adaptive_config.learning_strategy == LearningStrategy.REINFORCEMENT:
            await self._q_learning_update(experience)
        elif self.adaptive_config.learning_strategy == LearningStrategy.IMITATION:
            await self._imitation_learning_update(experience)
        elif self.adaptive_config.learning_strategy == LearningStrategy.EVOLUTION:
            await self._evolutionary_update(experience)
        elif self.adaptive_config.learning_strategy == LearningStrategy.GRADIENT_DESCENT:
            await self._gradient_descent_update(experience)

        # Experience replay
        if self.adaptive_config.experience_replay and len(self.experience_buffer) > self.adaptive_config.replay_batch_size:
            await self._experience_replay()

        # Update performance metrics
        self._update_performance_metrics(experience)

        # Decay exploration rate
        self._decay_exploration_rate()

    async def _q_learning_update(self, experience: Experience) -> None:
        """Update policy using Q-learning."""
        learning_rate = self.adaptive_config.learning_rate
        discount_factor = self.adaptive_config.discount_factor

        # Convert states to feature vectors
        state_features = np.array([experience.state[feature] for feature in self.state_features])
        next_state_features = np.array([experience.next_state[feature] for feature in self.state_features])

        # Get action index
        action_index = next(i for i, a in enumerate(self.action_space) if a.value == experience.action)

        # Calculate TD error
        current_q = await self._get_q_value(experience.state, ActionType(experience.action))

        # Get max Q-value for next state
        max_next_q = float('-inf')
        for action in self.action_space:
            q_val = await self._get_q_value(experience.next_state, action)
            if q_val > max_next_q:
                max_next_q = q_val

        target_q = experience.reward + discount_factor * max_next_q
        td_error = target_q - current_q

        # Update weights
        if len(self.policy_params.weights.shape) == 2:
            self.policy_params.weights[:, action_index] += learning_rate * td_error * state_features
        else:
            # Update action value directly
            self.action_values[experience.action] += learning_rate * td_error

    async def _imitation_learning_update(self, experience: Experience) -> None:
        """Update policy using imitation learning."""
        # Store successful experiences from role models for imitation
        if experience.reward > 0:
            self.imitation_buffer.append(experience)

    async def _evolutionary_update(self, experience: Experience) -> None:
        """Update policy using evolutionary strategy."""
        # Add experience to fitness evaluation
        # This would be used in population-based updates
        pass

    async def _gradient_descent_update(self, experience: Experience) -> None:
        """Update policy using gradient descent."""
        learning_rate = self.adaptive_config.learning_rate

        # Calculate policy gradient
        state_features = np.array([experience.state[feature] for feature in self.state_features])

        # Update policy parameters
        self.policy_params.weights += learning_rate * experience.reward * state_features
        self.policy_params.bias += learning_rate * experience.reward

    async def _experience_replay(self) -> None:
        """Perform experience replay learning."""
        if len(self.experience_buffer) < self.adaptive_config.replay_batch_size:
            return

        # Sample random batch of experiences
        batch_size = min(self.adaptive_config.replay_batch_size, len(self.experience_buffer))
        batch = self.rng.sample(list(self.experience_buffer), batch_size)

        # Learn from each experience in the batch
        for experience in batch:
            await self._q_learning_update(experience)

    def _update_performance_metrics(self, experience: Experience) -> None:
        """Update performance tracking metrics."""
        # Add reward to history
        self.reward_history.append(experience.reward)

        # Update metrics
        self.performance_metrics.total_reward += experience.reward

        if len(self.reward_history) > 0:
            self.performance_metrics.average_reward = sum(self.reward_history) / len(self.reward_history)

        # Update success rate
        if experience.reward > 0:
            experience.success = True

        recent_successes = sum(1 for exp in list(self.experience_buffer)[-20:] if exp.success)
        self.performance_metrics.success_rate = recent_successes / min(20, len(self.experience_buffer))

        # Update learning progress (rate of improvement)
        if len(self.reward_history) >= 2:
            recent_avg = sum(list(self.reward_history)[-10:]) / min(10, len(self.reward_history))
            older_avg = sum(list(self.reward_history)[-20:-10]) / min(10, len(self.reward_history))
            self.performance_metrics.learning_progress = recent_avg - older_avg

    def _decay_exploration_rate(self) -> None:
        """Decay exploration rate over time."""
        current_rate = self.policy_params.exploration_rate
        decay_rate = self.adaptive_config.exploration_decay
        min_rate = self.adaptive_config.min_exploration_rate

        new_rate = max(min_rate, current_rate * decay_rate)
        self.policy_params.exploration_rate = new_rate

    def _calculate_local_disorder(self, neighbors: Dict[CellID, CellData]) -> float:
        """Calculate local sorting disorder."""
        if not neighbors:
            return 0.0

        inversions = 0
        total_pairs = 0

        neighbor_list = list(neighbors.values())
        for i, cell_a in enumerate(neighbor_list):
            for cell_b in neighbor_list[i+1:]:
                total_pairs += 1
                if cell_a.sort_value > cell_b.sort_value:
                    if cell_a.position.x < cell_b.position.x:
                        inversions += 1

        return inversions / max(1, total_pairs)

    def _get_recent_average_reward(self) -> float:
        """Get recent average reward."""
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)

    def _calculate_exploration_bonus(self) -> float:
        """Calculate exploration bonus for curiosity-driven learning."""
        # Bonus for exploring new state-action combinations
        state_action_count = len(set((exp.state.get('position_x', 0), exp.action) for exp in self.experience_buffer))
        total_experiences = max(1, len(self.experience_buffer))
        diversity_bonus = state_action_count / total_experiences

        return diversity_bonus

    def _estimate_swap_benefit(self, neighbor_data: CellData) -> float:
        """Estimate benefit of swapping with a neighbor."""
        # Compare sort values
        my_value = self.current_data.sort_value
        their_value = neighbor_data.sort_value

        # Benefit if we would be better sorted after swap
        position_benefit = 0.0
        if self.current_data.position.x < neighbor_data.position.x and my_value > their_value:
            position_benefit = abs(my_value - their_value)
        elif self.current_data.position.x > neighbor_data.position.x and my_value < their_value:
            position_benefit = abs(my_value - their_value)

        return position_benefit

    async def _update_role_models(self) -> None:
        """Update list of role models for imitation learning."""
        if not self.adaptive_config.imitation_enabled:
            return

        neighbors = self.get_neighbors()
        imitation_threshold = self.adaptive_config.imitation_threshold

        # Evaluate neighbor performance
        for neighbor_id, neighbor_data in neighbors.items():
            # Estimate neighbor's performance (simplified)
            performance_score = self._estimate_neighbor_performance(neighbor_data)

            if performance_score > imitation_threshold:
                self.role_models[neighbor_id] = performance_score
            elif neighbor_id in self.role_models:
                # Remove poor performers
                del self.role_models[neighbor_id]

        # Limit number of role models
        if len(self.role_models) > 5:
            # Keep only the best ones
            best_models = sorted(self.role_models.items(), key=lambda x: x[1], reverse=True)[:5]
            self.role_models = dict(best_models)

    def _estimate_neighbor_performance(self, neighbor_data: CellData) -> float:
        """Estimate performance of a neighbor for imitation learning."""
        # Simple heuristic based on neighbor's apparent success
        # In a real implementation, this would observe actual performance

        # Assume neighbors with good sort positioning are performing well
        neighbors_of_neighbor = self.get_neighbors()  # Approximation
        local_disorder = 0.0

        if neighbors_of_neighbor:
            # Estimate their local disorder
            for other_neighbor in neighbors_of_neighbor.values():
                if other_neighbor.cell_id != neighbor_data.cell_id:
                    if neighbor_data.sort_value > other_neighbor.sort_value:
                        if neighbor_data.position.x < other_neighbor.position.x:
                            local_disorder += 1

        # Performance is inverse of disorder
        performance = 1.0 - (local_disorder / max(1, len(neighbors_of_neighbor)))
        return performance

    async def _imitate_model(self, model_id: CellID) -> Optional[CellAction]:
        """Imitate the behavior of a role model."""
        # This is a simplified imitation - in practice, would observe model's actions
        neighbors = self.get_neighbors()

        if model_id not in neighbors:
            return None

        model_data = neighbors[model_id]

        # Simple imitation: try to match their behavior
        # If they're in a good position, try to get closer
        distance = self.current_data.distance_to(model_data)

        if distance > 2.0:
            # Move closer to role model
            direction_x = model_data.position.x - self.current_data.position.x
            direction_y = model_data.position.y - self.current_data.position.y
            magnitude = math.sqrt(direction_x**2 + direction_y**2)

            if magnitude > 0:
                move_distance = min(self.behavior_config.max_movement_distance, distance * 0.5)
                target_x = self.current_data.position.x + (direction_x / magnitude) * move_distance
                target_y = self.current_data.position.y + (direction_y / magnitude) * move_distance

                return create_move_action(
                    self.cell_id,
                    self.last_update_timestep,
                    Position(target_x, target_y)
                )

        return None

    async def _handle_message(self, message) -> None:
        """Handle adaptive learning specific messages."""
        await super()._handle_message(message)

        if message.message_type == "performance_feedback":
            # Receive performance feedback from coordinator
            performance_score = message.data.get("performance", 0.0)

            # Use as reward signal
            if self.experience_buffer:
                self.experience_buffer[-1].reward += performance_score

        elif message.message_type == "role_model_update":
            # Update role model information
            model_data = message.data.get("role_models", {})
            for model_id, performance in model_data.items():
                self.role_models[CellID(model_id)] = performance

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.performance_metrics

    def get_policy_parameters(self) -> PolicyParameters:
        """Get current policy parameters."""
        return self.policy_params

    def get_learning_progress(self) -> float:
        """Get learning progress score."""
        return self.performance_metrics.learning_progress

    def get_exploration_rate(self) -> float:
        """Get current exploration rate."""
        return self.policy_params.exploration_rate

    def get_experience_count(self) -> int:
        """Get number of experiences collected."""
        return len(self.experience_buffer)

    def get_role_models(self) -> Dict[CellID, float]:
        """Get current role models and their performance scores."""
        return self.role_models.copy()