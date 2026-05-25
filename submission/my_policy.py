"""
Policy for the heuristic dispatcher submission.

The observation builder writes the dispatcher's recommended action into
obs[0..4] as a one-hot vector. The policy argmaxes that block and masks
against obs[5..9] (the action mask) for safety.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class DispatcherPolicy:
    def __init__(self):
        pass

    def reset(self):
        pass

    def act_many(
        self, handles: List[int], observations: List[Any], **kwargs
    ) -> Dict[int, int]:
        return {h: self.act(obs) for h, obs in zip(handles, observations)}

    def act(self, observation: Any, **kwargs) -> int:
        obs = np.asarray(observation, dtype=np.float32)
        recommended = obs[:5]
        mask = obs[5:10] if obs.shape[0] >= 10 else np.ones(5, dtype=np.float32)
        scored = np.where(mask > 0.5, recommended, -np.inf)
        if not np.any(np.isfinite(scored)):
            return 0  # DO_NOTHING fallback
        return int(np.argmax(scored))


MyPolicy = DispatcherPolicy
