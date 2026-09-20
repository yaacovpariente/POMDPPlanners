Maze
====


These three environments test memory of a noisy, single-use cue. The map is
known and movement is deterministic; only the rewarding goal side is hidden.
Entering either goal ends the episode, but only the correct goal counts as
success. The cue names that side with probability ``cue_accuracy``; later
observations are ``empty``.

Formal definition
-----------------

All three share one model and differ only in :math:`A` and how a move is
resolved. Let :math:`G` be the walkable cells, :math:`\chi \in G` the cue
cell, and :math:`\gamma_L, \gamma_R \in G` the two goal cells.

**State space.** Position, the hidden goal side, and the cue's delivery phase:

.. math::

   s = (x,\; y,\; \mathrm{side},\; \varphi)

.. math::

   S = \mathcal{P} \times \{\textsf{L}, \textsf{R}\} \times
   \{\textsf{UNSEEN}, \textsf{EMITTING}, \textsf{CONSUMED}\}

with :math:`\mathcal{P} = G` for the discrete variants and
:math:`\mathcal{P} \subseteq \mathbb{R}^2` the walkable region for the
continuous one. Carrying :math:`\varphi` *in the state* rather than as a flag
on the environment object is what keeps the problem Markov: a planner
resamples transitions from arbitrary states out of order, and an episode flag
living on ``self`` would be written by the search as well as by the world.

**Action space.**

.. math::

   A = \begin{cases}
     \{\textsf{up}, \textsf{down}, \textsf{left}, \textsf{right}\}
       & \texttt{DiscreteMazePOMDP},\ \texttt{TMazePOMDP} \\
     \{d \in \mathbb{R}^2 : \lVert d \rVert_2 \leq \texttt{max\_step\_size}\}
       & \texttt{ContinuousMazePOMDP}
   \end{cases}

A longer displacement is rescaled to the cap rather than rejected.

**Transition model.** Deterministic, and a goal is absorbing:

.. math::

   T(s' \mid s, a) = \mathbb{1}[s' = f(s, a)], \qquad
   f(s, a) = s \ \text{ for } s \in S_T

The position update refuses illegal moves without moving the agent:

.. math::

   (x', y') = \begin{cases}
     (x, y) + \Delta_a & \text{the move is legal} \\
     (x, y) & \text{otherwise (wall collision)}
   \end{cases}

A discrete step is legal when the target cell is walkable. A continuous step
is legal only when the **whole swept segment** stays inside the walkable
region — checking only the endpoint would let a step jump a wall. The goal
side never changes. The cue phase advances on every action, including one a
wall refused:

.. math::

   \varphi' = \begin{cases}
     \textsf{EMITTING} & \varphi = \textsf{UNSEEN}
       \text{ and the step crosses } \chi \\
     \textsf{CONSUMED} & \varphi = \textsf{EMITTING} \\
     \varphi & \text{otherwise}
   \end{cases}

That middle branch is what makes the cue **single-use**: it is consumed by
whatever action follows it, so standing still on the cue cell cannot re-read
it and a revisit never yields a second reading.

**Observation model.**

.. math::

   \Omega = \{\textsf{left\_cue},\; \textsf{right\_cue},\; \textsf{empty}\}

.. math::

   O(\textsf{left\_cue} \mid s', \cdot) &= \begin{cases}
     \alpha & \varphi' = \textsf{EMITTING},\ \mathrm{side} = \textsf{L} \\
     1 - \alpha & \varphi' = \textsf{EMITTING},\ \mathrm{side} = \textsf{R} \\
     0 & \varphi' \neq \textsf{EMITTING}
   \end{cases} \\
   O(\textsf{empty} \mid s', \cdot) &= \mathbb{1}[\varphi' \neq \textsf{EMITTING}]

with :math:`\alpha` = ``cue_accuracy`` :math:`\in [0.5, 1]`, and
:math:`\textsf{right\_cue}` mirrored. The action does not enter. There is
deliberately **no** wall observation: three of the four actions bump into a
wall almost everywhere on a corridor, so such a reading would leak position
information the task is not about.

Under the uniform prior, one :math:`\textsf{left\_cue}` at
:math:`\alpha = 0.9` moves the belief to :math:`0.9 / 0.1`, and every
subsequent :math:`\textsf{empty}` leaves it exactly there. That flat stretch
is the whole task: a planner that does not track a belief has nothing left to
turn on when it reaches the junction.

**Reward function.** The terminal payout **replaces** the step cost rather
than stacking with it, so no two terms ever add:

.. math::

   R(s, a) = \begin{cases}
     0 & s \in S_T \\
     +\texttt{goal\_reward} & \text{the step enters } \gamma_{\mathrm{side}} \\
     -\texttt{wrong\_goal\_penalty} & \text{the step enters the other goal} \\
     -\texttt{step\_penalty} & \text{otherwise, wall collisions included}
   \end{cases}

The best achievable return is therefore exactly the goal reward discounted by
the number of steps taken to reach it.

**Initial belief.** Position and cue phase known, goal side a coin flip:

.. math::

   b_0\big((x_0, y_0, \textsf{L}, \textsf{UNSEEN})\big) =
   b_0\big((x_0, y_0, \textsf{R}, \textsf{UNSEEN})\big) = \tfrac{1}{2}

The opening observation is fixed at :math:`\textsf{empty}`. The cue is emitted
by *crossing* the cue cell and the runner only observes after an action, so a
reading here would hand the agent the answer before it had moved.

**Discount.** :math:`\gamma` = ``discount_factor``, default :math:`0.95`.

**Terminal set.** Either goal, correct or not:

.. math::

   S_T = \{s : (x, y) \in \gamma_L \cup \gamma_R\}

Both are reported, separately: a planner that guesses wrong at the junction
and one that never reaches it fail for opposite reasons.

DiscreteMazePOMDP
-----------------

``DiscreteMazePOMDP`` uses integer cell positions in a generated maze. The
four actions, ``up``, ``down``, ``left`` and ``right``, move one cell; a wall
blocks the move. ``maze_width``, ``maze_height``, ``maze_seed`` and
``loop_fraction`` control the layout.

.. figure:: ../images/discrete_maze_visualization.gif
   :alt: DiscreteMazePOMDP recorded episode with cell guides, cue and two goals.
   :width: 100%

   DiscreteMazePOMDP: one-cell moves through a generated maze.

ContinuousMazePOMDP
-------------------

``ContinuousMazePOMDP`` uses real-valued positions and displacement actions
``[dx, dy]``, capped in length by ``max_step_size``. Collision checks cover
the whole movement path. With the same layout settings as DiscreteMazePOMDP,
it uses the same maze geometry.

.. figure:: ../images/continuous_maze_visualization.gif
   :alt: ContinuousMazePOMDP recorded episode with a continuous position trail.
   :width: 100%

   ContinuousMazePOMDP: bounded displacements through a generated maze.

Belief
------

``create_environment_belief`` returns a
``MazeVectorizedWeightedParticleBelief`` for either maze. The hidden state is
one bit -- which goal pays -- so the update is cheap per particle and the Python
loop around it is the whole cost; ``DiscreteMazeVectorizedUpdater`` and
``ContinuousMazeVectorizedUpdater`` do that loop's work in NumPy instead.

Both reproduce the environment's event rule rather than approximating it: the
discrete one by the same lookup table the environment builds, the continuous one
by the segment test written as array algebra. A belief that walked through walls
the world refuses would be searching a different maze. The continuous updater's
positions agree with the environment's to within its cell tolerance rather than
bit for bit, because it stops a step at the tolerance-widened cell boundary the
same code uses to decide membership.

TMazePOMDP
----------

``TMazePOMDP`` uses a T-shaped corridor and the same four one-cell actions as
the discrete maze. ``stem_length`` sets the distance to the junction and
``arm_length`` sets the distance from the junction to each endpoint. The
agent must remember the cue while walking up the stem, then choose an arm.

.. figure:: ../images/t_maze_visualization.gif
   :alt: TMazePOMDP recorded episode on a T-shaped corridor with two endpoints.
   :width: 100%

   TMazePOMDP: remember the cue until the left-or-right choice at the junction.

The images above are package-generated golden visualizations, copied unchanged
from the environment visualization test fixtures. They illustrate recorded
histories, not planner performance comparisons.

Create a visualization
----------------------

All three variants share a renderer for reviewing recorded episodes. The GIF is 1200 by 800 pixels and advances every 500 ms.

.. code-block:: python

   from pathlib import Path
   from POMDPPlanners.environments.maze_pomdp import MazeVisualizer

   # Use the environment and History returned by the simulation workflow.
   MazeVisualizer(environment).create_visualization(
       episode.history, Path("maze-review.gif")
   )

Read a frame
------------

The red dot is the recorded position; the red line joins earlier positions.
Ivory cells are walkable. Slate regions and their outlines are walls. Discrete
mazes have cell guides; continuous positions are drawn without rounding.

The blue triangle marks the cue cell. The open square marks the start. L and R
identify the candidate goals. The green star shows the true goal for the human
observer; this is hidden from the policy.

Amber rings show recorded position particles, sized by weight. Endpoint tints
and the left/right bars show the recorded probability of each goal. A missing
belief is labeled ``unavailable``.

The action belongs to the displayed state. Last observation and last reward
come from the preceding transition, so the first frame has neither. Total
reward is the undiscounted sum of completed transitions. The final state needs
a terminal record, as supplied by the simulation workflow.

The public ``MazeVisualizer`` import and legacy ``TMazeVisualizer`` alias are
unchanged. The renderer isolates its style from the caller's matplotlib theme.
