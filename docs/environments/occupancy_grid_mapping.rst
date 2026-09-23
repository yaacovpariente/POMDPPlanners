Occupancy grid mapping
======================

.. episode-viewer:: traces/occupancy_grid_mapping.json

   One real episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

``OccupancyGridMappingPOMDP`` models a robot mapping a hidden static grid with
known pose and noisy range scans. The default world is 10 by 10 cells with a
boundary wall and three random rectangular obstacles. The robot starts at the
centre, facing north.

The reward is the entropy removed from the robot's own map, so the planner has
to choose where to drive to see the cells it is still unsure about. Integer
pose is a design simplification; particle-based MCTS also supports continuous
states.

What the agent sees and does
----------------------------

- **State** — a ``float64`` vector of length ``4 + 2 * num_cells + num_beams``
  (228 at the defaults): step, row, column, heading, the hidden true
  occupancy, the observation-derived map log-odds, and the last noisy scan.
- **Actions** (discrete) — ``OccupancyGridAction``: ``FORWARD = 0`` moves one
  cell along the heading; ``TURN_LEFT = 1`` and ``TURN_RIGHT = 2`` rotate 90
  degrees in place.
- **Observations** (continuous) — a ``float64`` vector
  ``[row, column, heading, ranges...]`` of length ``3 + num_beams``. Pose is
  exact and integer; the ``num_beams`` ranges (24 by default) are noisy, in
  cells.

Formal definition
-----------------

The environment is the POMDP :math:`\langle S, A, \Omega, T, O, R, b_0, \gamma
\rangle`. Let the grid have :math:`H` rows and :math:`W` columns,
:math:`C = HW` cells, and :math:`K` = ``num_beams``.

**State space.** The state is *augmented*: it carries the robot's own map
estimate and the scan drawn this step, alongside the hidden world.

.. math::

   s = \big(\underbrace{t}_{\text{step}},\;
   \underbrace{(r, c, d)}_{\text{pose}},\;
   \underbrace{m}_{\text{true map}},\;
   \underbrace{g}_{\text{log-odds}},\;
   \underbrace{z}_{\text{scan}}\big)

.. math::

   S = \mathbb{Z}_{\geq 0} \times
   \big(\{0..H{-}1\} \times \{0..W{-}1\} \times \{0,1,2,3\}\big) \times
   \{0,1\}^{C} \times [-L, L]^{C} \times \mathbb{R}^{K}

of length :math:`4 + 2C + K`, with :math:`L` = ``log_odds_clamp``. The hidden
part is :math:`m`; the robot's map :math:`g` is a *statistic the agent
computed*, not a fact about the world, and is carried in the state only so the
reward can be a function of it.

**Action space**

.. math::

   A = \{\textsf{forward},\; \textsf{turn\_left},\; \textsf{turn\_right}\}
     = \{0, 1, 2\}

**Observation space.** The exact pose and one range per beam:

.. math::

   \Omega = \{0..H{-}1\} \times \{0..W{-}1\} \times \{0,1,2,3\}
   \times \mathbb{R}^{K}

**Transition model.** Three stages.

*Motion.* Turning is deterministic, :math:`d' = (d \mp 1) \bmod 4`.
A forward move into an occupied or out-of-bounds cell is blocked; an otherwise
valid move fails with probability :math:`p_f` = ``move_failure_probability``:

.. math::

   \Pr[(r', c') = \mathrm{fwd}(r, c, d)] = 1 - p_f, \qquad
   \Pr[(r', c') = (r, c)] = p_f

The true map :math:`m` never changes.

*Scan.* Ray-cast the :math:`K` beams of the fan from the new pose against
:math:`m` to get noise-free ranges :math:`\bar{z}_k`, then draw

.. math::

   z_k \sim \mathcal{N}(\bar{z}_k, \sigma^2) \quad\text{or}\quad
   \mathcal{N}_{\geq 0}(\bar{z}_k, \sigma^2)

independently per beam, per ``range_noise_model``, with :math:`\sigma` =
``range_noise_std_cells``.

*Map update.* The scan is folded into :math:`g` by the inverse sensor
model. Under the default ``NearestCellLogOddsUpdateRule``, with
:math:`\ell_{\text{occ}} = \operatorname{logit}(\texttt{hit\_probability})` and
:math:`\ell_{\text{free}} = \operatorname{logit}(\texttt{miss\_probability})`:

.. math::

   g'_j = \mathrm{clip}\Big(g_j + \sum_{k=1}^{K}
   \big(\ell_{\text{occ}}\,\mathbb{1}[j = h_k]
   + \ell_{\text{free}}\,\mathbb{1}[j \prec h_k]\big)
   + \ell_{\text{free}}\,\mathbb{1}[j = (r', c')],\;
   -L,\; L\Big)

where :math:`h_k` is the cell beam :math:`k` stopped in and :math:`j \prec h_k`
means :math:`j` lies on the ray before it. A beam reading at or above the
maximum range frees its whole in-grid ray and marks nothing occupied. All terms
are summed and clamped **once**, never per term — clamping per term would make
the beam order matter.

Finally :math:`t' = t + 1`.

**Observation model.** Because the scan is drawn in the transition and stored
in :math:`s'`, the observation kernel is a point mass:

.. math::

   o = (r', c', d', z'), \qquad
   O(o \mid s', a) = \mathbb{1}[o = (r', c', d', z')]

This is a deliberate reformulation, not a claim that the robot sees
everything: all the stochasticity has been moved into :math:`T`. The quantity
a filter actually needs is the predictive density, which integrates the
discrete motion outcome against the per-beam range law:

.. math::

   p(o \mid s, a) = \sum_{u} \Pr[u \mid s, a]\,
   \mathbb{1}\big[(r', c', d') = \mathrm{pose}(u)\big]
   \prod_{k=1}^{K} p\big(z_k \mid \bar{z}_k(u), \sigma\big)

exposed as ``predictive_observation_log_probability``.

**Reward function.** Entropy removed from the robot's map, in bits. Treating
cells as independent Bernoulli variables — the assumption occupancy-grid
mapping is defined under — write

.. math::

   \mathcal{H}(g) = \sum_{j=1}^{C} H_{\text{bin}}\big(\mathrm{sigmoid}(g_j)\big),
   \qquad \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

with :math:`H_{\text{bin}}` the binary entropy in bits, so one unknown cell is worth
exactly :math:`1.0` and a wholly unknown grid exactly :math:`C`. Then

.. math::

   R(s, a, s') = \mathcal{H}(g) - \mathcal{H}(g')
   - \texttt{step\_cost}

.. note::

   Called without :math:`s'` — as a belief-space planner's expected reward
   does — the environment returns the numerical expectation
   :math:`\mathcal{H}(g) - \mathbb{E}[\mathcal{H}(g')]` over eight
   fixed antithetic quadrature points per beam, not a fresh sample. It is an
   approximation and uses no global randomness. Neither quantity is posterior
   whole-map information gain.

**Initial belief.** The robot's pose is known; the map is not:

.. math::

   b_0 = \mathbb{1}\big[(t, r, c, d) = (0, r_0, c_0, d_0)\big] \otimes
   \mathrm{Prior}(m) \otimes \mathbb{1}[g = 0] \otimes \mathbb{1}[z = 0]

where :math:`\mathrm{Prior}(m)` places ``num_obstacles`` random rectangles of
side at most ``max_obstacle_size``, plus the boundary wall when
``has_boundary_wall``, keeping the start cell free. :math:`g = 0` is the
uninformative prior :math:`p = \tfrac{1}{2}` everywhere, so
:math:`\mathcal{H}(g_0) = C`. The opening observation is the start pose
with zero ranges, a sentinel never passed to the map update.

**Discount.** :math:`\gamma` = ``discount_factor``, default :math:`0.95`.

**Terminal set.** Map resolved, or budget spent:

.. math::

   S_T = \{s : \mathcal{H}(g) \leq q\, \mathcal{H}(g_0)\}
   \;\cup\; \{s : t \geq \texttt{max\_steps}\}

with :math:`q` = ``entropy_threshold_fraction``. At the default
:math:`q = 0.25` that is a quarter of a bit per cell on average, roughly
96 % certainty per cell.

State and observation contract
------------------------------

The hidden map constrains motion and produces nominal ranges. Range noise is
drawn once in the transition, then the same scan is stored, used for mapping,
and revealed by the observation.

Repeated observation calls on one successor reveal the same stored scan. The
augmented observation kernel is a point mass. The predictive density
``p(observation | prior state, action)`` combines the per-beam range density
with the probability of the observed motion outcome.

The initial observation contains known pose and zero range placeholders.
It is a sentinel before any scan and is never applied to the map.

Range noise model
-----------------

``range_noise_model`` selects the per-beam range law that
``range_noise_std_cells`` parametrises. Both laws are centred on the noise-free
range :math:`\bar{z}` and neither truncates above the maximum range.

``gaussian`` (the default) is the unbounded normal :math:`\mathcal{N}(\bar{z}, \sigma^2)`. It is
what every result produced before the option existed used, and its seeded
draws are unchanged, so earlier runs still reproduce. A reading below zero is
possible under it; the inverse model treats such a reading as a hit in the
first ray cell.

``truncated_normal`` is the same normal conditioned on ``z >= 0``:

.. math::

   p(z \mid \bar{z}, \sigma) = \frac{\mathcal{N}(z;\, \bar{z}, \sigma^2)}{F(\bar{z} / \sigma)}
   \quad \text{for } z \ge 0, \qquad 0 \text{ otherwise,}

with :math:`F` the standard normal CDF.

It is a renormalised density, not a clamp: no probability mass is moved onto
zero. The normaliser :math:`F(\bar{z} / \sigma)` depends on the nominal range, so two
map particles that predict different ranges for one beam are normalised
differently, and both filters carry that term per beam and per particle. A
negative reading has zero likelihood under this law. Sampling inverts the
truncated distribution exactly, with one uniform per beam, so no tail is
dropped and no draw is rejected. With the default ``sigma = 0.35`` a beam at
full range sits ten standard deviations above zero and the two laws agree to
machine precision; they differ where an obstacle is within a few standard
deviations of the robot.

.. code-block:: python

   env = OccupancyGridMappingPOMDP(range_noise_model="truncated_normal")

The mode is part of ``config_id`` and of both filters' identities, so results
cached under one law are never reused for the other. The sensor contract
version is 3 from this option onwards.

Mapping and reward
------------------

Log-odds start at zero, or occupancy probability 0.5. The inverse update reads
only the previous log-odds and observed pose/ranges. A reading below maximum
range selects the nearest ray-cell centre; cells before it get free evidence,
the selected cell gets occupied evidence, and cells after it remain unchanged.
Negative readings select the first valid cell. Ties select the nearer cell.
A reading at or above maximum range marks the full in-grid ray free.
The robot's observed cell also receives free evidence. Log-odds are clamped.

That rule is the default, not a fixed property of the environment. It is
``NearestCellLogOddsUpdateRule``, built from ``hit_probability``,
``miss_probability`` and ``log_odds_clamp``, and it is what every result
produced before the rule became pluggable used. The ``update_rule`` constructor
argument replaces it, and one rule object serves the scalar transition, the
batched particle kernels and both rewards, so the three cannot disagree::

    from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
        OccupancyGridMappingPOMDP,
        ProbabilityOccupancyUpdateRule,
    )


    class DampedOddsRule(ProbabilityOccupancyUpdateRule):
        """Half the usual confidence per sighting."""

        def update_probabilities(self, probabilities, free_counts, occupied_counts):
            ratio = 0.4**free_counts * 2.5**occupied_counts
            scaled = probabilities * ratio
            return scaled / (1.0 - probabilities + scaled)

        def parameters(self):
            return super().parameters()


    env = OccupancyGridMappingPOMDP(update_rule=DampedOddsRule())

Subclass ``ProbabilityOccupancyUpdateRule`` to write the update on occupancy
probabilities, as the literature states it: the base class classifies the scan
into per-cell free and occupied sighting counts, converts ``L`` to ``p`` before
your method and back after it, and applies the clamp. Subclass
``OccupancyUpdateRule`` instead to work directly on log-odds, or to change how
a scan is classified rather than only the arithmetic on the counts; implement
its batched method as well if the rule runs inside a planner's belief update,
since the default loops over the scalar one.

A rule becomes read-only the moment an environment or an updater adopts it.
Its parameters are hashed into that object's ``config_id`` once, so allowing
``env.update_rule.occupied_log_odds = 0.5`` afterwards would map under a
different law while the cache still answered to the old identity. Build a new
rule and a new environment instead.

``parameters()`` is both the rule's contribution to the environment's
``config_id`` and the keyword arguments it is rebuilt from, so a subclass must
accept every key it reports and report every parameter that changes the update.
The default rule contributes nothing, because the three settings that define it
are already in the configuration -- so a default environment's ``config_id`` is
what it always was, and every cached episode stays valid. Any other rule does
contribute, so its results can never be served from a cache filled under a
different rule.

A true hit exactly at maximum range and a miss have identical range laws.
They therefore produce identical updates for the same reading. Without an
observed hit flag, this ambiguity cannot be removed. Range noise can place
an apparent hit beyond a real obstacle or before it; the mapper follows the
measurement, not hidden truth.

Simulation reward is the realised decrease in the observed inverse map's
summed binary entropy, minus ``step_cost``. The environment declares
``reward_requires_next_state=True`` so the runner supplies that realised map.
Its ``reward_range`` is ``(-num_cells - step_cost, num_cells - step_cost)``:
both entropies lie in ``[0, num_cells]``, so one step can neither gain nor lose
more than the whole grid.
When a planner requests reward without a successor, the explicit fallback uses
eight fixed antithetic unit-normal points per motion outcome, mapped through
the selected range law, to approximate the expected decrease. This
deterministic integration consumes no simulation RNG, but has integration
error. It updates each hypothetical map with the same
observation-based rule as the actual map.

This is an exploration surrogate inspired by entropy-reduction objectives,
not exact posterior information gain. The inverse map treats cells and beam
increments as independent. Repeated correlated beams can create confidence in
an incorrect map. Low entropy measures confidence, not map accuracy.

An episode completes when the observed inverse map's entropy reaches
``entropy_threshold_fraction`` of its initial value (25% by default).
``max_steps`` (40 by default) ends an unfinished episode. Reward and completion
use the inverse estimate; neither substitutes the true occupancy grid.

Filtering and limits
--------------------

Use one of the environment's two whole-map filters with PFT_DPW. Ordinary
``get_initial_belief`` returns a bootstrap filter which cannot condition this
stored continuous scan correctly. No core planner changes are needed.

Both filters install the observed scan and its map update in every
particle. They weight whole-map hypotheses by the predictive range density
of the selected law and exact motion probability, with no epsilon floor for
impossible poses.
Routine resampling is disabled to retain low-weight hypotheses. If every
particle contradicts observed motion, bounded prior replay tests up to 4096
fresh maps against the entire observation history and resamples surviving
weighted maps. It raises if that search finds no support.

``OccupancyGridMappingVectorizedBelief`` is the default from
``create_environment_belief`` and from ``BeliefType.VECTORIZED_PARTICLE``. It
scores all particles with one batched ray cast and applies one shared
inverse-sensor delta to all of them, so an update costs a few array
operations rather than two environment calls per particle. It gives the same
particles, weights, history and restart count as the scalar filter for the
same seed. ``OccupancyGridMappingBelief`` is the scalar reference, available
as ``BeliefType.PARTICLE``. Its updater also exposes the batched generative
transition and point-mass observation likelihood that the shared vectorized
updater interface expects, but the filter does not use them: a freshly drawn
scan matches the observed one with probability zero.

The environment's ``reward_batch`` uses the same kernels. A belief-space
planner asks for the expected reward of every particle at every node it
expands, and without a successor each scalar call integrates eight
hypothetical scans; batching those is where most of PFT_DPW's decision time
goes, so it is what makes the planner faster with either filter.

Thirty particles is a QA resource choice, not a guarantee against degeneracy.
Reweighting cannot create missing maps. Prior replay is finite importance
sampling and may leave only one effective hypothesis. Inspect effective sample
size, unique maps, restarts and map error alongside completion. Unweighted
rejection filters are unsuitable for exact matching of continuous scans.

There is no torch vectorized or C++ model. VOPP is unsupported. Scalar PFT_DPW
uses the environment and the whole-map filters shown above.

Metrics
-------

``task_completion_rate`` reports threshold crossing. ``ended_by_goal``,
``ended_by_failure`` and ``ended_by_timeout`` report episode endings; failure
is always zero. Progress metrics include residual entropy and resolved-cell
fraction. ``average_obstacle_collisions`` counts blocked moves;
``average_successful_translations`` counts successful moves, including revisits.
It is not a unique-cell count.

Visualization
-------------

Runs also write a GIF of each episode through ``cache_visualization``. Its left
panel shows the observation-derived inverse map. The middle shows weighted
occupancy marginals over whole-map particles. The right shows the hidden true
map for review. Panels depict the state before the captioned action; the
caption's reward is the realised inverse-map entropy reduction.

Sensor contract versions 2 and 3 each changed cache identity; old results and
GIFs describe the earlier behavior. The golden GIF is rendered in the default Gaussian mode
and is unchanged by the truncated option.

Minimal example
---------------

.. code-block:: python

   from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
       OccupancyGridMappingPOMDP,
   )
   from POMDPPlanners.utils.belief_factory import create_environment_belief

   env = OccupancyGridMappingPOMDP()
   belief = create_environment_belief(env, n_particles=30)

See also
--------

- :class:`POMDPPlanners.environments.occupancy_grid_mapping_pomdp.OccupancyGridMappingPOMDP`
- :doc:`index` — the full catalog.
