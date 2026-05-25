"""
PPO trainer for the FastTreeObs + ActorCritic baseline.

Usage:
    python reinforcement-learning/train.py \
        --total-steps 2_000_000 \
        --num-agents 5 \
        --width 30 --height 30 \
        --save-path reinforcement-learning/checkpoint.pt

Drop the resulting checkpoint into submission/checkpoint.pt and rebuild the
Docker image.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import (
    MalfunctionParameters,
    ParamMalfunctionGen,
)
from flatland.envs.step_utils.states import TrainState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_observation_builder import FastTreeObsBuilder
from my_policy import ActorCritic


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(args, seed: int) -> RailEnv:
    obs_builder = FastTreeObsBuilder(max_depth=args.tree_depth, with_action_mask=True)
    malfunction = ParamMalfunctionGen(
        MalfunctionParameters(
            malfunction_rate=1 / 1000,
            min_duration=15,
            max_duration=50,
        )
    )
    env = RailEnv(
        width=args.width,
        height=args.height,
        rail_generator=sparse_rail_generator(
            max_num_cities=args.num_cities,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=2,
            seed=seed,
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=args.num_agents,
        obs_builder_object=obs_builder,
        malfunction_generator=malfunction,
        random_seed=seed,
    )
    return env


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    obs: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = ActorCritic(
        obs_size=FastTreeObsBuilder.OBSERVATION_DIM,
        n_actions=FastTreeObsBuilder.ACTION_MASK_SIZE,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        checkpoint_path=args.resume_from if args.resume_from else None,
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    env = make_env(args, seed=args.seed)
    obs_dict, _info = env.reset()

    obs_size = FastTreeObsBuilder.OBSERVATION_DIM
    n_actions = FastTreeObsBuilder.ACTION_MASK_SIZE

    global_step = 0
    iteration = 0
    episode_returns: List[float] = []
    episode_return = 0.0
    t_start = time.time()

    while global_step < args.total_steps:
        iteration += 1
        rollout = Rollout()
        # Track per-handle obs to attribute next-step rewards correctly.
        steps_in_rollout = 0

        while steps_in_rollout < args.rollout_steps:
            handles = env.get_agent_handles()
            obs_batch = np.stack(
                [np.asarray(obs_dict[h], dtype=np.float32) for h in handles]
            )
            features = obs_batch[:, :obs_size]
            masks = obs_batch[:, obs_size : obs_size + n_actions]

            with torch.no_grad():
                logits, values = model(torch.from_numpy(features).to(device))
                logits = logits.masked_fill(
                    torch.from_numpy(masks).to(device) < 0.5, float("-inf")
                )
                # Fail-safe: if every action is masked, allow DO_NOTHING.
                all_masked = torch.isinf(logits).all(dim=-1)
                if all_masked.any():
                    logits[all_masked, 0] = 0.0
                dist = Categorical(logits=logits)
                actions_t = dist.sample()
                logprobs_t = dist.log_prob(actions_t)

            actions = actions_t.cpu().numpy()
            action_dict = {h: int(actions[i]) for i, h in enumerate(handles)}

            next_obs_dict, rewards, dones, _info = env.step(action_dict)

            # Per-agent transition.
            for i, h in enumerate(handles):
                rollout.obs.append(features[i])
                rollout.masks.append(masks[i])
                rollout.actions.append(int(actions[i]))
                rollout.logprobs.append(float(logprobs_t[i].item()))
                rollout.values.append(float(values[i].item()))
                rollout.rewards.append(float(rewards[h]))
                rollout.dones.append(1.0 if dones["__all__"] else 0.0)
                episode_return += float(rewards[h])

            steps_in_rollout += len(handles)
            global_step += len(handles)

            if dones["__all__"]:
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs_dict, _info = env.reset()
            else:
                obs_dict = next_obs_dict

        # Bootstrap value for last state.
        handles = env.get_agent_handles()
        obs_batch = np.stack(
            [np.asarray(obs_dict[h], dtype=np.float32) for h in handles]
        )
        features = obs_batch[:, :obs_size]
        with torch.no_grad():
            _, last_values = model(torch.from_numpy(features).to(device))
        last_value = float(last_values.mean().item())

        advantages, returns = compute_gae(
            rollout.rewards, rollout.values, rollout.dones,
            last_value, args.gamma, args.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Tensors.
        b_obs = torch.from_numpy(np.stack(rollout.obs)).to(device)
        b_masks = torch.from_numpy(np.stack(rollout.masks)).to(device)
        b_actions = torch.from_numpy(np.asarray(rollout.actions, dtype=np.int64)).to(device)
        b_logprobs = torch.from_numpy(np.asarray(rollout.logprobs, dtype=np.float32)).to(device)
        b_returns = torch.from_numpy(returns).to(device)
        b_advantages = torch.from_numpy(advantages).to(device)

        n_samples = b_obs.shape[0]
        idx = np.arange(n_samples)
        for _epoch in range(args.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n_samples, args.minibatch_size):
                mb = idx[start : start + args.minibatch_size]
                mb_t = torch.from_numpy(mb).to(device)

                logits, values = model(b_obs[mb_t])
                logits = logits.masked_fill(b_masks[mb_t] < 0.5, float("-inf"))
                all_masked = torch.isinf(logits).all(dim=-1)
                if all_masked.any():
                    logits[all_masked, 0] = 0.0
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(b_actions[mb_t])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - b_logprobs[mb_t])
                surr1 = ratio * b_advantages[mb_t]
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * b_advantages[mb_t]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, b_returns[mb_t])
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        if iteration % args.log_every == 0:
            recent = episode_returns[-20:] if episode_returns else [0.0]
            elapsed = time.time() - t_start
            sps = global_step / max(elapsed, 1e-6)
            print(
                f"iter={iteration} step={global_step} "
                f"ep_return_mean={np.mean(recent):.2f} "
                f"loss={loss.item():.3f} ent={entropy.item():.3f} "
                f"sps={sps:.0f}",
                flush=True,
            )

        if iteration % args.save_every == 0:
            save_checkpoint(model, args.save_path)

    save_checkpoint(model, args.save_path)
    print(f"done. checkpoint saved to {args.save_path}")


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({"model": model.state_dict()}, path)


def parse_args():
    p = argparse.ArgumentParser()
    # Env.
    p.add_argument("--width", type=int, default=30)
    p.add_argument("--height", type=int, default=30)
    p.add_argument("--num-agents", type=int, default=5)
    p.add_argument("--num-cities", type=int, default=3)
    p.add_argument("--tree-depth", type=int, default=3)
    # PPO.
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--rollout-steps", type=int, default=4096)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    # Model.
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    # I/O.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=str, default="reinforcement-learning/checkpoint.pt")
    p.add_argument("--resume-from", type=str, default="")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
