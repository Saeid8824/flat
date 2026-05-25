"""
Observation builder for the heuristic dispatcher submission.

The observation is small by design: the policy is currently a deterministic
"read the dispatcher's recommendation" wrapper. The same layout will be
extended for the BC/PPO stages with additional features (priority rank,
slack, etc.) starting from index 10.

Layout (per agent):
    [0..4]   one-hot recommended action from HeuristicDispatcher
    [5..9]   action mask (5 dims; DO_NOTHING/L/F/R/STOP)
    [10]     normalised priority rank in [0, 1]  (0 = highest priority)
    [11]     clipped slack ratio in [-1, 1]
    [12]     state flag: 1 if on map, 0 otherwise
    [13]     malfunction countdown / max_episode_steps in [0, 1]
"""

from __future__ import annotations

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.step_utils.states import TrainState

from submission.dispatcher import HeuristicDispatcher


OBS_DIM = 14


class DispatcherObservationBuilder(ObservationBuilder):
    def __init__(self):
        super().__init__()
        self.dispatcher = HeuristicDispatcher()

    def set_env(self, env):
        super().set_env(env)
        self.dispatcher.set_env(env)

    def reset(self):
        self.dispatcher.reset()

    def get(self, handle: int) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action = int(self.dispatcher.act(handle))
        if 0 <= action < 5:
            obs[action] = 1.0

        # Action mask (mirrors FastTreeObs logic).
        agent = self.env.agents[handle]
        mask = obs[5:10]
        mask[0] = 1.0  # DO_NOTHING always valid
        if agent.state in (TrainState.DONE, TrainState.DONE_REMOVED, TrainState.WAITING):
            pass
        elif agent.position is None:
            mask[2] = 1.0  # READY_TO_DEPART -> MOVE_FORWARD enters the grid
        else:
            mask[4] = 1.0  # STOP_MOVING always valid on map
            pt = self.env.rail.get_transitions((agent.position, agent.direction))
            d = int(agent.direction)
            mask[1] = float(pt[(d - 1) % 4])
            mask[2] = float(pt[d])
            mask[3] = float(pt[(d + 1) % 4])

        # Extra features for downstream RL.
        rank = self.dispatcher.priority_rank(handle)
        n_active = max(self.env.get_num_agents(), 1)
        obs[10] = (rank / n_active) if rank >= 0 else 1.0

        slack = self.dispatcher.slack(handle)
        max_steps = max(int(self.env._max_episode_steps), 1)
        # Slack can be large positive or negative; clip to [-1, 1] of episode length.
        obs[11] = float(np.clip(slack / max_steps, -1.0, 1.0))

        on_map = agent.state in (
            TrainState.MOVING,
            TrainState.STOPPED,
            TrainState.MALFUNCTION,
        )
        obs[12] = 1.0 if on_map else 0.0

        mh = getattr(agent, "malfunction_handler", None)
        if mh is not None:
            mf = int(getattr(mh, "malfunction_down_counter", 0))
            obs[13] = float(np.clip(mf / max_steps, 0.0, 1.0))

        return obs


MyObservationBuilder = DispatcherObservationBuilder
