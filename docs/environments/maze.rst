Maze visualization
==================

``DiscreteMazePOMDP`` and ``ContinuousMazePOMDP`` share a renderer for reviewing
recorded episodes. The GIF is 1200 by 800 pixels and advances every 500 ms.

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
