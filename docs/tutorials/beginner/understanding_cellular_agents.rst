Understanding Cellular Agents
============================

This tutorial introduces the fundamental concept of cellular agents - the computational models that represent individual cells in our morphogenesis simulations.

What Are Cellular Agents?
-------------------------

**Cellular agents** are autonomous software entities that model individual biological cells. Each agent:

* Operates independently, making its own decisions
* Communicates with neighboring agents (like cells communicating through signaling)
* Follows simple behavioral rules that can lead to complex collective behaviors
* Maintains its own state (position, cell type, internal variables)

This approach allows us to study how thousands of individual cells coordinate to create tissues, organs, and complex biological patterns.

Biological Inspiration
----------------------

In real biology, cells don't have a "master plan" - they follow simple local rules:

**Cell Communication:**
   Cells send chemical signals to their neighbors and respond to signals they receive.

**Cell Adhesion:**
   Cells prefer to stick to certain types of neighbors (like cells tend to cluster with similar cells).

**Cell Migration:**
   Cells can move in response to chemical gradients or physical forces.

**Cell Division and Death:**
   Cells reproduce when conditions are right and die when damaged or no longer needed.

Our cellular agents capture these biological principles in computational form.

Anatomy of a Cellular Agent
---------------------------

Let's examine the key components of a cellular agent in our platform:

.. code-block:: python

   from core.agents.cell_agent import AsyncCellAgent
   from core.data.types import Position, CellType, CellState

   class ExampleCell(AsyncCellAgent):
       """Example cellular agent showing key components."""

       def __init__(self, cell_id: str, position: Position, cell_type: CellType):
           super().__init__(cell_id, position, cell_type)

           # Agent state
           self.position = position
           self.cell_type = cell_type
           self.energy = 100.0
           self.age = 0

           # Behavioral parameters
           self.movement_speed = 1.0
           self.communication_range = 2.0
           self.adhesion_preferences = {
               'same_type': 0.8,    # Prefer similar cells
               'different_type': 0.2  # Avoid different cells
           }

**Key Components:**

**Identity:**
   * ``cell_id``: Unique identifier for the agent
   * ``cell_type``: What kind of cell this is ('epithelial', 'mesenchymal', etc.)

**Spatial Properties:**
   * ``position``: Current location in simulation space
   * ``movement_speed``: How quickly the cell can move
   * ``communication_range``: How far the cell can "sense" neighbors

**Internal State:**
   * ``energy``: Metabolic state of the cell
   * ``age``: How long the cell has existed
   * Custom variables specific to your research question

**Behavioral Parameters:**
   * ``adhesion_preferences``: Which cell types this cell prefers as neighbors
   * Algorithm-specific parameters

Agent Behaviors
---------------

Cellular agents can exhibit various biologically-inspired behaviors:

**1. Sorting Behavior**
   Cells organize themselves by type, similar to how different tissue types separate during development.

.. code-block:: python

   async def sorting_behavior(self):
       """Cells prefer neighbors of the same type."""
       neighbors = await self.get_neighbors()

       # Count similar and different neighbors
       similar_neighbors = sum(1 for n in neighbors if n.cell_type == self.cell_type)
       total_neighbors = len(neighbors)

       if total_neighbors > 0:
           similarity_ratio = similar_neighbors / total_neighbors

           # Move if too many different neighbors
           if similarity_ratio < 0.5:
               target_position = await self.find_better_location()
               await self.move_toward(target_position)

**2. Adaptive Behavior**
   Cells change their behavior based on local environmental conditions.

.. code-block:: python

   async def adaptive_behavior(self):
       """Cells adapt to local chemical gradients."""
       local_gradient = await self.sense_chemical_gradient()

       # Adapt behavior based on gradient
       if local_gradient > 0.7:
           self.behavior = 'proliferative'  # Start dividing
           self.movement_speed = 0.5       # Slow down
       elif local_gradient < 0.3:
           self.behavior = 'migratory'     # Start moving
           self.movement_speed = 2.0       # Speed up
       else:
           self.behavior = 'maintaining'   # Stay put

**3. Pattern Formation**
   Cells coordinate to create spatial patterns like stripes or spots.

.. code-block:: python

   async def pattern_formation_behavior(self):
       """Cells follow morphogen gradients to form patterns."""
       morphogen_level = await self.sense_morphogen()
       neighbors = await self.get_neighbors()

       # Activate different behaviors based on morphogen concentration
       if morphogen_level > 0.8:
           self.cell_type = 'activated'
           await self.secrete_signal('activation_signal', strength=1.0)
       elif morphogen_level < 0.2:
           self.cell_type = 'inhibited'
           await self.secrete_signal('inhibition_signal', strength=1.0)

Communication Between Agents
----------------------------

Cellular agents communicate through several mechanisms:

**1. Direct Neighbor Communication**

.. code-block:: python

   async def communicate_with_neighbors(self):
       """Send messages to adjacent cells."""
       neighbors = await self.get_neighbors(distance=1)

       for neighbor in neighbors:
           message = {
               'sender_id': self.cell_id,
               'message_type': 'status_update',
               'content': {
                   'cell_type': self.cell_type,
                   'energy_level': self.energy,
                   'stress_level': self.calculate_stress()
               }
           }
           await neighbor.receive_message(message)

**2. Chemical Signaling**

.. code-block:: python

   async def chemical_signaling(self):
       """Secrete and sense chemical signals."""
       # Secrete signals based on internal state
       if self.energy > 80:
           await self.secrete_chemical('growth_factor', concentration=0.5)

       if self.stress_level > 0.7:
           await self.secrete_chemical('stress_hormone', concentration=0.8)

       # Sense chemicals in the environment
       local_chemicals = await self.sense_environment()

       # Respond to chemical cues
       if local_chemicals.get('growth_factor', 0) > 0.3:
           self.proliferation_rate += 0.1

       if local_chemicals.get('stress_hormone', 0) > 0.5:
           self.movement_speed += 0.2  # Move away from stress

**3. Physical Interactions**

.. code-block:: python

   async def physical_interactions(self):
       """Handle physical forces and constraints."""
       neighbors = await self.get_neighbors(distance=1)

       # Calculate crowding pressure
       crowding_pressure = len(neighbors) / self.max_neighbors

       if crowding_pressure > 0.8:
           # Too crowded - either move or reduce activity
           empty_spots = await self.find_empty_neighbors()
           if empty_spots:
               await self.move_toward(random.choice(empty_spots))
           else:
               self.proliferation_rate *= 0.5  # Reduce division

Your First Cellular Agent
-------------------------

Let's create a simple cellular agent for a sorting experiment:

.. code-block:: python

   import asyncio
   from core.agents.cell_agent import AsyncCellAgent
   from core.data.types import Position, CellType

   class SimpleSortingCell(AsyncCellAgent):
       """A basic cell that demonstrates sorting behavior."""

       def __init__(self, cell_id: str, position: Position, cell_type: CellType):
           super().__init__(cell_id, position, cell_type)
           self.happiness = 0.0  # How satisfied is this cell with its neighbors?

       async def update(self):
           """Main update loop for the cellular agent."""
           # Step 1: Assess current situation
           await self.calculate_happiness()

           # Step 2: Decide on action
           if self.happiness < 0.5:
               await self.attempt_to_move()

           # Step 3: Update internal state
           self.age += 1

       async def calculate_happiness(self):
           """Calculate how satisfied this cell is with its neighbors."""
           neighbors = await self.get_neighbors()

           if not neighbors:
               self.happiness = 0.5  # Neutral if no neighbors
               return

           # Count neighbors of same type
           same_type_neighbors = sum(1 for n in neighbors if n.cell_type == self.cell_type)
           total_neighbors = len(neighbors)

           # Happiness increases with proportion of similar neighbors
           self.happiness = same_type_neighbors / total_neighbors

       async def attempt_to_move(self):
           """Try to find a better location with more similar neighbors."""
           current_position = self.position
           best_position = current_position
           best_happiness = self.happiness

           # Check potential moves
           for dx in [-1, 0, 1]:
               for dy in [-1, 0, 1]:
                   if dx == 0 and dy == 0:
                       continue

                   new_position = Position(
                       current_position.x + dx,
                       current_position.y + dy
                   )

                   # Check if position is available and calculate potential happiness
                   if await self.is_position_available(new_position):
                       potential_happiness = await self.calculate_potential_happiness(new_position)

                       if potential_happiness > best_happiness:
                           best_position = new_position
                           best_happiness = potential_happiness

           # Move if we found a better location
           if best_position != current_position:
               await self.move_to(best_position)
               self.happiness = best_happiness

Running Your Cellular Agent
---------------------------

Here's how to run a simulation with your custom cellular agents:

.. code-block:: python

   import asyncio
   from core.coordination.coordinator import DeterministicCoordinator
   from core.data.types import Position, CellType

   async def run_sorting_simulation():
       """Run a simulation with custom sorting cells."""

       # Create coordinator
       coordinator = DeterministicCoordinator(grid_size=(20, 20))

       # Create cellular agents
       cells = []
       for i in range(50):
           # Create mix of cell types
           cell_type = CellType('red' if i < 25 else 'blue')
           position = Position(
               x=i % 20,
               y=i // 20
           )

           cell = SimpleSortingCell(
               cell_id=f"cell_{i}",
               position=position,
               cell_type=cell_type
           )

           cells.append(cell)
           await coordinator.add_agent(cell)

       # Run simulation
       print("Starting morphogenesis simulation...")
       for step in range(100):
           await coordinator.step()

           if step % 10 == 0:
               # Calculate average happiness
               total_happiness = sum(cell.happiness for cell in cells)
               avg_happiness = total_happiness / len(cells)
               print(f"Step {step}: Average happiness = {avg_happiness:.3f}")

       print("Simulation complete!")
       return coordinator

   # Run the simulation
   if __name__ == "__main__":
       coordinator = asyncio.run(run_sorting_simulation())

Expected Output:

.. code-block:: text

   Starting morphogenesis simulation...
   Step 0: Average happiness = 0.234
   Step 10: Average happiness = 0.456
   Step 20: Average happiness = 0.623
   Step 30: Average happiness = 0.751
   Step 40: Average happiness = 0.834
   Step 50: Average happiness = 0.889
   Step 60: Average happiness = 0.912
   Step 70: Average happiness = 0.934
   Step 80: Average happiness = 0.945
   Step 90: Average happiness = 0.951
   Simulation complete!

The increasing happiness shows that cells are successfully sorting themselves by type!

Key Concepts Summary
--------------------

**Cellular Agents Are:**
   * Autonomous - they make their own decisions
   * Reactive - they respond to their local environment
   * Interactive - they communicate with neighbors
   * Adaptive - they can change behavior over time

**Important Design Principles:**
   * **Local Rules**: Each cell only knows about its immediate surroundings
   * **Simple Behaviors**: Individual behaviors are simple, but collective behavior is complex
   * **Emergent Properties**: Interesting patterns arise from many cells interacting
   * **Biological Realism**: Behaviors are inspired by real cellular biology

**Common Patterns:**
   * Sense → Decide → Act: The basic agent loop
   * Neighbor Communication: How cells coordinate
   * State Management: Tracking internal cell properties
   * Environmental Response: Adapting to local conditions

Exercises
---------

**Exercise 1: Modify Happiness Calculation**
   Change how cells calculate happiness. Instead of just counting same-type neighbors, make cells prefer a specific ratio (e.g., 60% same type, 40% different type).

**Exercise 2: Add Cell Energy**
   Give cells an energy system where moving costs energy, and cells with low energy can't move. How does this affect sorting?

**Exercise 3: Implement Cell Division**
   Add a division behavior where happy cells occasionally create new cells of the same type. How does this change the population dynamics?

**Exercise 4: Create a New Cell Type**
   Add a third cell type that acts as a "mediator" - it's happy when surrounded by mixed neighbors rather than similar ones.

Next Steps
----------

Now that you understand cellular agents, you're ready to:

* :doc:`first_sorting_experiment` - Run a complete sorting experiment with analysis
* :doc:`basic_visualization` - See your agents in action with animations
* :doc:`../intermediate/adaptive_cell_behavior` - Explore more complex behaviors

**For Biology Students:**
   Research real examples of cell sorting in development (like germ layer formation) and think about how our agents model these processes.

**For Computer Science Students:**
   Explore how cellular agents relate to other agent-based modeling paradigms and multi-agent systems.

**For Researchers:**
   Consider how you might modify cellular agents to model the specific biological system you're studying.