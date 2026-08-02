# SPDX-License-Identifier: MIT

"""Concrete CARLA model with a kinematic bicycle transition under the control preset.

:class:`KinematicCarlaModelPOMDP` replaces the identity-placeholder transition of
:class:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_factored_model_pomdp.FactoredCarlaModelPOMDP`
with a real ego-motion model: it propagates the ego ``[x, y, yaw, vx, vy, lat,
heading_err]`` forward one tick under the selected ``(throttle, steer, brake)`` control
using a point-mass longitudinal model plus a bicycle yaw model, and closes the range on
tracked agents by the distance the ego travelled. The factored observation model, reward,
and terminal check are inherited unchanged.

This is what gives a planner a gradient toward accelerating: because throttle now visibly
increases the along-lane speed the reward rewards, POMCPOW picks controls that actually
move the car (the identity placeholder made every action look motionless).

Classes:
    KinematicCarlaModelPOMDP: Factored CARLA model with a kinematic ego transition.
"""

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_factored_model_pomdp import (
    FactoredCarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    DEFAULT_MAX_TRACKED_AGENTS,
    EGO_STATE_WIDTH,
    driving_quality_reward,
)


class KinematicCarlaModelPOMDP(FactoredCarlaModelPOMDP):
    """Factored CARLA model whose transition is a kinematic bicycle propagation.

    Attributes:
        dt: Integration step (seconds); must match the world's ``fixed_delta_seconds``.
        wheelbase: Bicycle-model wheelbase (m) mapping steer to yaw rate.
        max_steer_angle: Steering command of 1.0 maps to this front-wheel angle (rad).
        accel: Longitudinal acceleration per unit throttle (m/s^2).
        brake_decel: Longitudinal deceleration per unit brake (m/s^2).
        drag: Linear speed-proportional deceleration coefficient (1/s).
        collision_gap: Forward ego-frame distance (m) within which a present agent
            ahead is treated as a predicted collision by :meth:`is_terminal`.
        collision_halfwidth: Lateral ego-frame half-corridor (m) within which a present
            agent ahead is treated as a predicted collision by :meth:`is_terminal`.
        safe_distance: Lead gap (m) at/above which the reward targets the full
            ``desired_speed``; the obstacle-aware target ramps down below it.
        stop_gap: Lead gap (m) at/below which the obstacle-aware target speed is zero;
            ``0.0`` keeps the flat ``desired_speed``.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
        ...     AGENT_SLOT_WIDTH, EGO_STATE_WIDTH)
        >>> env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)
        >>>
        >>> width = EGO_STATE_WIDTH + env.max_tracked_agents * AGENT_SLOT_WIDTH
        >>> state = np.zeros(width)
        >>> throttle_action = 0  # (0.5, 0.0, 0.0) cruise straight
        >>> next_state = env.sample_next_state(state, throttle_action)
        >>>
        >>> bool(next_state[3] > 0.0)  # throttle produced forward velocity
        True
    """

    def __init__(
        self,
        discount_factor: float,
        dt: float = 0.05,
        action_presets: Optional[Sequence[Tuple[float, float, float]]] = None,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        wheelbase: float = 2.8,
        max_steer_angle: float = 0.6,
        accel: float = 3.0,
        brake_decel: float = 8.0,
        drag: float = 0.05,
        collision_gap: float = 5.0,
        collision_halfwidth: float = 1.2,
        safe_distance: float = 12.0,
        stop_gap: float = 0.0,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the kinematic CARLA model.

        Args:
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            dt: Integration step (s); match the world's ``fixed_delta_seconds``.
            action_presets: Discrete ``(throttle, steer, brake)`` triples.
            max_tracked_agents: Number of fixed agent slots in state/observation.
            wheelbase: Bicycle-model wheelbase (m).
            max_steer_angle: Front-wheel angle (rad) at steering command 1.0.
            accel: Longitudinal acceleration per unit throttle (m/s^2).
            brake_decel: Longitudinal deceleration per unit brake (m/s^2).
            drag: Linear speed-proportional deceleration coefficient (1/s).
            collision_gap: Forward ego-frame distance (m) within which a present agent
                ahead counts as a predicted collision (terminal).
            collision_halfwidth: Lateral ego-frame half-corridor (m) within which a
                present agent ahead counts as a predicted collision (terminal).
            safe_distance: Forward ego-frame gap (m) at/above which the full ``desired_speed``
                is targeted; below it the obstacle-aware target ramps down toward ``stop_gap``.
            stop_gap: Forward ego-frame gap (m) at/below which the obstacle-aware target speed
                is zero (the ego should be stopped). ``0.0`` keeps the flat ``desired_speed``
                (no obstacle-aware speed shaping).
            name: Environment identifier. Defaults to the class name.
            **kwargs: Forwarded to the factored observation/reward parent (e.g.
                ``desired_speed``, ``pose_std``, ``collision_penalty``).
        """
        self.dt = dt
        self.wheelbase = wheelbase
        self.max_steer_angle = max_steer_angle
        self.accel = accel
        self.brake_decel = brake_decel
        self.drag = drag
        self.collision_gap = collision_gap
        self.collision_halfwidth = collision_halfwidth
        self.safe_distance = safe_distance
        self.stop_gap = stop_gap
        super().__init__(
            discount_factor=discount_factor,
            action_presets=action_presets,
            max_tracked_agents=max_tracked_agents,
            name=name,
            **kwargs,
        )

    # ── Reward (driving quality with an obstacle-aware target speed) ──────
    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        """Driving-quality reward whose target speed adapts to the nearest lead obstacle.

        The bare parent reward tracks a *fixed* ``desired_speed`` and only charges the
        collision once an agent is inside the terminal box — a cliff that cannot be braked
        for at speed, while a large fixed penalty instead freezes the ego in traffic. This
        override follows Roach's obstacle-aware desired speed: the target equals the full
        ``desired_speed`` when the lead gap is at least ``safe_distance``, ramps linearly to
        zero at ``stop_gap``, and is zero closer in. The ego is thus rewarded for driving
        when the road is clear and for slowing as an obstacle nears — including the
        lidar/camera obstacle fused into the agent slots — without a separate penalty term
        that traps it at a standstill. ``stop_gap == 0`` keeps the flat parent behaviour.
        """
        resulting = np.asarray(next_state if next_state is not None else state, dtype=float)
        _, steer, _ = self.action_presets[int(action)]
        return driving_quality_reward(
            resulting,
            steer,
            self.is_terminal(resulting),
            self._obstacle_aware_desired_speed(resulting),
            self.out_lane_thresh,
            self.collision_penalty,
        )

    def _obstacle_aware_desired_speed(self, state: np.ndarray) -> float:
        if self.stop_gap == 0.0:
            return self.desired_speed
        gap = self._nearest_lead_gap(state)
        if gap >= self.safe_distance:
            return self.desired_speed
        if gap <= self.stop_gap:
            return 0.0
        ramp = (gap - self.stop_gap) / (self.safe_distance - self.stop_gap)
        return self.desired_speed * ramp

    def _nearest_lead_gap(self, state: np.ndarray) -> float:
        rows = self._state_agent_rows(state)
        gaps = [
            float(rows[slot, 1])
            for slot in range(self.max_tracked_agents)
            if rows[slot, 0] == 1.0
            and rows[slot, 1] > 0.0
            and abs(rows[slot, 2]) < self.collision_halfwidth
        ]
        return min(gaps) if gaps else float("inf")

    # ── Terminal (predicted collision with a tracked agent) ──────────────
    def is_terminal(self, state: Any) -> bool:
        """Whether a present agent occupies the ego's footprint just ahead.

        Because this model *does* predict ego and agent motion, it can foresee running
        into the vehicle ahead. Any present agent slot within ``collision_gap`` metres
        forward and ``collision_halfwidth`` metres laterally (ego frame) is treated as a
        collision, which the inherited reward turns into the terminal
        ``collision_penalty`` (:func:`driving_quality_reward`).
        """
        rows = self._state_agent_rows(np.asarray(state, dtype=float))
        for slot in range(self.max_tracked_agents):
            if rows[slot, 0] != 1.0:
                continue
            rel_x, rel_y = float(rows[slot, 1]), float(rows[slot, 2])
            if 0.0 <= rel_x < self.collision_gap and abs(rel_y) < self.collision_halfwidth:
                return True
        return False

    # ── Transition (kinematic bicycle under the control preset) ──────────
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        control = self.action_presets[int(action)]
        next_state = self._propagate(np.asarray(state, dtype=float), control)
        if n_samples == 1:
            return next_state
        return np.stack([next_state.copy() for _ in range(n_samples)])

    def sample_next_state_batch(self, states: Any, action: Any) -> np.ndarray:
        control = self.action_presets[int(action)]
        states_arr = np.asarray(states, dtype=float)
        return np.stack([self._propagate(state, control) for state in states_arr])

    def _propagate(self, state: np.ndarray, control: Tuple[float, float, float]) -> np.ndarray:
        next_state = state.copy()
        next_state[:EGO_STATE_WIDTH] = self._propagate_ego(state[:EGO_STATE_WIDTH], control)
        self._advance_agents(next_state, float(next_state[3]), float(next_state[4]))
        return next_state

    def _propagate_ego(self, ego: np.ndarray, control: Tuple[float, float, float]) -> np.ndarray:
        throttle, steer, brake = control
        x, y, yaw_deg, vel_x, vel_y, lateral, heading_err = ego
        yaw = np.radians(yaw_deg)
        speed = float(np.hypot(vel_x, vel_y))

        longitudinal_accel = throttle * self.accel - brake * self.brake_decel - self.drag * speed
        speed_next = max(0.0, speed + longitudinal_accel * self.dt)
        yaw_rate = (speed_next / self.wheelbase) * np.tan(steer * self.max_steer_angle)
        yaw_next = yaw + yaw_rate * self.dt

        vx_next = speed_next * np.cos(yaw_next)
        vy_next = speed_next * np.sin(yaw_next)
        heading_err_next = heading_err + yaw_rate * self.dt
        lateral_next = lateral + speed_next * np.sin(heading_err_next) * self.dt
        return np.array(
            [
                x + vx_next * self.dt,
                y + vy_next * self.dt,
                np.degrees(yaw_next),
                vx_next,
                vy_next,
                lateral_next,
                heading_err_next,
            ]
        )

    def _advance_agents(self, state: np.ndarray, vx_next: float, vy_next: float) -> None:
        ego_speed = float(np.hypot(vx_next, vy_next))
        rows = self._state_agent_rows(state)
        for slot in range(self.max_tracked_agents):
            if rows[slot, 0] != 1.0:
                continue
            # Close the ego-frame range by the net longitudinal closing speed.
            rows[slot, 1] += (rows[slot, 4] - ego_speed) * self.dt
        state[EGO_STATE_WIDTH:] = rows.reshape(-1)
