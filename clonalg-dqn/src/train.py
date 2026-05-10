import gymnasium as gym
import numpy as np
import csv
import os
import torch
from collections import deque
from tqdm import tqdm

from src.dqn import DQNAgent
from src.clonalg_buffer import CLONALGBuffer
from src.utils import set_seed


# ── Buffer aleatorio simple (línea base) ─────────────────────────────────
class RandomReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done, td_error=None):
        self.buffer.append((obs, action, reward, next_obs, float(done)))

    def sample(self, batch_size):
        idx   = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return {
            "obs":      np.array(obs),
            "actions":  np.array(actions),
            "rewards":  np.array(rewards),
            "next_obs": np.array(next_obs),
            "dones":    np.array(dones),
            "indices":  None,
        }

    def __len__(self):
        return len(self.buffer)


# ── Entrenamiento ─────────────────────────────────────────────────────────
def train(seed=0, n_episodes=300, batch_size=64,
          buffer_capacity=2000, min_buffer=200,
          use_clonalg=False, lr=1e-4,
          clone_rate=0.5, mutation_rate=0.1, suppression_threshold=0.95,
          log_dir="logs", run_tag="baseline"):

    set_seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    env       = gym.make("LunarLander-v3")
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    agent = DQNAgent(obs_dim, n_actions, lr=lr, device=device)

    if use_clonalg:
        buffer = CLONALGBuffer(
            capacity=buffer_capacity,
            clone_rate=clone_rate,
            mutation_rate=mutation_rate,
            suppression_threshold=suppression_threshold,
        )
    else:
        buffer = RandomReplayBuffer(capacity=10_000)

    log_path = os.path.join(log_dir, f"{run_tag}_seed{seed}.csv")
    with open(log_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "return", "epsilon", "loss", "mean_affinity"])

        episode_returns = []

        for ep in tqdm(range(n_episodes), desc=f"[{run_tag}] seed={seed}"):
            obs, _       = env.reset(seed=seed + ep)
            total_reward = 0.0
            ep_loss      = []
            ep_affinity  = []

            while True:
                action                              = agent.select_action(obs)
                next_obs, reward, term, trunc, _    = env.step(action)
                done                                = term or trunc

                buffer.add(obs, action, reward, next_obs, done)
                obs           = next_obs
                total_reward += reward

                if len(buffer) >= min_buffer:
                    batch            = buffer.sample(batch_size)
                    loss, td_errors  = agent.update(batch)
                    ep_loss.append(loss)

                    if use_clonalg and batch["indices"] is not None:
                        buffer.update_affinity(batch["indices"], td_errors)
                        if buffer.mean_affinity_log:
                            ep_affinity.append(buffer.mean_affinity_log[-1])

                if done:
                    break

            episode_returns.append(total_reward)
            mean_loss     = np.mean(ep_loss)     if ep_loss     else 0.0
            mean_affinity = np.mean(ep_affinity) if ep_affinity else 0.0
            writer.writerow([ep, total_reward, agent.epsilon,
                             mean_loss, mean_affinity])

    env.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Retorno medio últimos 50 ep: {np.mean(episode_returns[-50:]):.1f}")
    return episode_returns


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["baseline", "clonalg", "both"],
                        default="baseline")
    parser.add_argument("--episodes",   type=int,   default=300)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--fast",       action="store_true")
    parser.add_argument("--capacity",   type=int,   default=2000)
    parser.add_argument("--clone_rate", type=float, default=0.5)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--tag",        type=str,   default=None)
    args = parser.parse_args()

    if args.fast:
        cfg = dict(n_episodes=100, buffer_capacity=500,
                   min_buffer=200, batch_size=32)
    else:
        cfg = dict(n_episodes=args.episodes, buffer_capacity=args.capacity,
                   min_buffer=500, batch_size=64)

    if args.mode in ("baseline", "both"):
        train(seed=args.seed, run_tag="baseline",
              use_clonalg=False, lr=args.lr, **cfg)

    if args.mode in ("clonalg", "both"):
        tag = args.tag or "clonalg"
        train(seed=args.seed, run_tag=tag,
              use_clonalg=True, lr=args.lr,
              clone_rate=args.clone_rate,
              **cfg)