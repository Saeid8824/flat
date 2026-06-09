import os
from pathlib import Path
import numpy as np

from typing import Optional, Tuple
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rewards import ECML2026Rewards
from reinforcement_learning.sampling.sampling_env_generator import sampling_env_generator


SAMPLING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampling")
DEFAULT_SCENARIO = os.path.join(SAMPLING_DIR, "level_0_scenario_1.pkl")

OUT_DIR = Path("reinforcement_learning/curriculum/envs")

scenes = ['scene_1', 'scene_4', 'scene_5']
n_agents = [[1,1], [10,10], [25,25]]
line_lengths = [2, 3, 4]

def create_curriculum_env(scenario_path: str = DEFAULT_SCENARIO, scene: Optional[str] = None, n_agents_range: Optional[Tuple[int, int]] = None, line_length: int = 2) -> RailEnv:
    """
    Create and persist curriculum environments.

    Parameters
    ----------
    scenario_path
        Path to a Flatland scenario pickle. Defaults to ``level_0_scenario_1.pkl``
        The pickle defines the rail grid topology; lines and timetables are re-sampled at every reset.
    scene
        scene from the ECML2026 competition. Should be one of "scene_1", "scene_2", "scene_3", "scene_4", "scene_5".
    n_agents_range
        Integer or range specifying the (range of) number of agents for the environment.
    line_length
        Maximum line length with minimum being 2 (start and destination).
    """

    env = RailEnvPersister.load_new(
        scenario_path,
        rewards=ECML2026Rewards(),
    )[0]

    env = sampling_env_generator(env, line_length=line_length, scene=scene)
    if n_agents_range is not None:
        _enable_varied_n_agents(env, n_agents_range)
        
    return env


def create_curriculum_envs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_order = 0
    for line_length in line_lengths:
        for n_agent in n_agents:
            for scene in scenes:
                env = create_curriculum_env(
                    scenario_path=DEFAULT_SCENARIO,
                    line_length=line_length,
                    scene=scene,
                    n_agents_range=(n_agent[0], n_agent[1]),
                )
                env.reset()
                n_actual = env.number_of_agents
                n_order_str = f"{n_order:02d}"
                out = OUT_DIR / f"{n_order_str}_{scene}_ll-{line_length}_a-{n_actual}.pkl"
                RailEnvPersister.save(env, str(out))
                n_order += 1
                print(f"saved {out}")


def _enable_varied_n_agents(env: RailEnv, n_agents_range: Tuple[int, int]) -> None:
    """
    Patch ``env.reset`` so each call randomizes ``number_of_agents``.

    Flatland's line generator is called on every reset and respects
    ``env.number_of_agents``, so this is all it takes to vary the count.
    """

    lo, hi = n_agents_range
    if lo < 1 or hi < lo:
        raise ValueError(f"Invalid n_agents_range={n_agents_range}; need 1 <= lo <= hi")

    original_reset = env.reset

    def reset_with_random_n(*args, **kwargs):
        env.number_of_agents = int(np.random.randint(lo, hi + 1))
        return original_reset(*args, **kwargs)

    env.reset = reset_with_random_n



if __name__ == "__main__":
    create_curriculum_envs()

