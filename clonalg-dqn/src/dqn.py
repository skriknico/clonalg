import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 target_update_freq=200, device="cpu"):
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.target_update_freq = target_update_freq
        self.device             = device
        self.step_count         = 0

        self.q_net      = QNetwork(obs_dim, n_actions).to(device)
        self.target_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

    def select_action(self, obs: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(obs_t).argmax(dim=1).item()

    def update(self, batch: dict):
        s  = torch.FloatTensor(batch["obs"]).to(self.device)
        a  = torch.LongTensor(batch["actions"]).to(self.device)
        r  = torch.FloatTensor(batch["rewards"]).to(self.device)
        r  = torch.clamp(r, -10.0, 10.0)
        s2 = torch.FloatTensor(batch["next_obs"]).to(self.device)
        d  = torch.FloatTensor(batch["dones"]).to(self.device)

        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(s2).max(dim=1).values
            target     = r + self.gamma * max_next_q * (1 - d)

        td_errors = (target - q_values).abs().detach().cpu().numpy()
        loss      = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item(), td_errors