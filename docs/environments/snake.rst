Snake
=====

``SnakePOMDP`` is the arcade game with the food hidden. The snake observes its
own body exactly and has to find food it cannot see, using a short-range vision
window and a noisy directional scent. The task is to grow to ``target_length``
without hitting a wall, hitting itself, or going too long without eating.

.. code-block:: python

   from POMDPPlanners.environments.snake_pomdp import SnakePOMDP, SnakeBelief

   env = SnakePOMDP(grid_size=12, target_length=10)
   belief = SnakeBelief.from_environment(env, n_particles=200)

The playable area is ``grid_size`` by ``grid_size`` cells. The walls are not
cells of the state: they sit just outside that area, so a head that steps off
the grid has hit one. The renderer draws them as a border around the board.

Ported from the MDP version in `snake-rl
<https://github.com/DragonWarrior15/snake-rl>`_, which is fully observable and
rewards the same events.

Actions, observations and rewards
---------------------------------

Three actions, relative to the current heading: ``0`` turn left, ``1`` go
straight, ``2`` turn right. There is no reversing action, because a snake that
turned back on itself would walk into its own neck every time. The heading is
not stored in the state; it is the direction from the second body cell to the
head.

An observation is a flat tuple of integers with three parts:

* the **body**, reported exactly, head first. The body update is deterministic,
  so this tells the agent nothing it could not have computed — it is there so
  the state is fully recoverable from the observation history;
* **seen**, the food's exact cell when the food is inside the vision window (a
  square of Chebyshev radius ``window_radius``, clipped to the grid) and the
  window fires, which happens with probability ``detection_probability``. There
  are no false positives, so a sighting is conclusive;
* **scent**, one of four diagonal quadrants relative to the head, correct with
  probability ``scent_accuracy``. Food that shares the head's row or column is
  compatible with two quadrants, which split that probability between them.

Every terminal state emits one fixed reading instead, so a win and a wall hit
are indistinguishable through the sensor.

The reward is ``+1`` for eating, ``-1`` for dying to a wall, to itself or to
starvation, and ``0`` otherwise. Winning happens by eating, so it pays the same
``+1`` and nothing more. Eating and dying cannot both happen on one step: the
food is never on a body cell, so the step that reaches it can be neither a wall
nor a self hit.

Dynamics
--------

The body update is deterministic. The action turns the heading, the head steps
into the next cell, and the tail is released — unless the step ate the food, in
which case the tail stays and the snake grows by one. That pair decides a rule
that is easy to get wrong: stepping into the cell the tail has just left is
legal, but stepping into the tail while eating is a self hit.

The only random part of a transition is where the food respawns after it is
eaten: uniformly over the cells the new body does not occupy.

Wall and self hits are checked first, then the win, then starvation. A snake
that reaches its target length by walking into a wall has still hit the wall.
``starvation_limit`` defaults to ``2 * grid_size ** 2``.

Belief
------

The body is known and the food is one cell, so the belief is a categorical
distribution over the grid. ``SnakeBelief`` carries it exactly. When the
observed length grows, the tracked food was eaten and the prior restarts as
uniform over the new body's free cells; otherwise the previous distribution
carries over with the new head cell ruled out — not eating proves the food is
not where the head has just arrived. Either prior is then multiplied by the
likelihood of the sighting and the scent, and normalised.

A generic particle filter runs on this environment too, but it is lossy here:
a sighting rules out every cell but one, and a filter that happened to hold no
particle there would floor every weight and resample cells the sensor has
already excluded.

Metrics
-------

``task_completion_rate`` is the fraction of episodes that reached
``target_length``. ``ended_by_goal``, ``ended_by_failure`` and
``ended_by_timeout`` partition the episodes between winning, dying and running
out of the runner's steps. ``wall_death_rate``, ``self_death_rate`` and
``starvation_death_rate`` say which death it was, which matters because they
call for opposite fixes — a planner that walks into walls is searching badly,
one that starves is not searching at all. ``max_steps_since_food`` reports how
close an episode came to starving even when it did not.

Recorded visualization
----------------------

.. image:: ../images/snake_visualization.gif
   :alt: Snake board with a green snake, a cyan vision window, fogged cells, an amber belief heatmap over the hidden food, and a side panel of score, length, step, scent and detection readouts.
   :width: 100%

The board carries four layers. The snake is drawn head first with a colour
gradient down its body and eyes pointing along the heading; the agent observes
it exactly, so it is drawn at full strength everywhere. The cyan outline is the
vision window, and the cells outside it are fogged — the fog is about the
*food*, not the body. The amber glow is the belief: the posterior probability
that the food is in that cell, scaled to the brightest cell of that frame,
because the posterior starts spread over the whole grid and collapses onto one
cell the moment the window fires. The apple is drawn only so a reviewer can
check the other layers against the truth, and is labelled as hidden from the
agent.

Each frame shows the state the step was taken *from*, and the panel reports the
reading the agent chose on — the previous step's observation — rather than the
one its action is about to produce. A death is marked with a red cross over the
head and a starvation with an amber hourglass.

This is a renderer fixture replayed from a fixed action sequence with real
transitions and real belief updates. It demonstrates the renderer, not planner
performance.

No vectorized model
-------------------

Snake has no torch vectorized generative model, so it cannot be run under VOPP.
``PFT_DPW`` runs on the scalar ``Environment`` API and is what the environment's
QA pass uses.
