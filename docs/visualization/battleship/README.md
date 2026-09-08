# Battleship episode view

Three equal-sized boards separate what the agent has learned, what its belief predicts, and the hidden fleet. The renderer accepts the same `List[StepData]` and GIF destination as before.

The left board shows only previously probed cells: a white cross on rust marks a hit, a dot on blue marks a miss, and unprobed cells remain pale. The amber ring marks the current action. The center board shows the recorded belief's occupancy marginal on a fixed 0–100% scale; cell labels round to whole percentages. A gray cell and dash mean unavailable belief. Above 10 rows, numbers are omitted to keep the heatmap readable.

The right board shows true occupancy. Each inset shape is one ship cell; shapes do not claim a ship identity or orientation. Orange crosses mark earlier probes. Row and column indices start at zero; the border and grid retain the environment's square geometry.

Every board shows the state before its own action. The bottom caption reports that action's observation and reward separately. The final no-action record removes the ring and says “Final recorded state”; it does not claim all ships were found. Frames last 1.4 seconds, with 2.4 seconds on the final record, and the GIF loops for review.

The reference GIF is [the Battleship test artifact](../../../POMDPPlanners/tests/test_environments/golden_visualizations/battleship_visualization.gif). It replays the existing deterministic fixture with seed 7, real environment transitions and real belief updates. It is a renderer fixture, not a planner performance result. Human review is still required.
