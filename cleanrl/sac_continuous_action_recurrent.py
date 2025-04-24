#!/usr/bin/env python3
"""
Recurrent SAC (LSTM) on top of CleanRL’s SAC-continuous implementation.
✓ partial-obs wrapper, ✓ frame-stacking, ✓ α-tuning, ✓ WandB logging
"""

import os, random, time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import FrameStack, FlattenObservation

torch.set_num_threads(1)


# ───────────────────────── recurrent encoder ──────────────────────────
class RecurrentEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers)
        self.hidden_dim, self.n_layers = hidden_dim, n_layers

    def forward(self, seq: torch.Tensor, hx=None):
        """
        seq : [seq_len, batch, feat]
        hx  : (h, c) each [n_layers, batch, hidden]
        """
        batch = seq.size(1)
        if hx is None or hx[0].size(1) != batch:
            hx = self.init_hidden(batch, seq.device)

        y, hx = self.lstm(seq, hx)
        hx = (hx[0].detach(), hx[1].detach())

        return y[-1], hx

    def init_hidden(self, batch_size: int, device):
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros_like(h0)
        return (h0, c0)


# ───────────────────────── argparse dataclass ─────────────────────────
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 1
    cuda: bool = True
    torch_deterministic: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False

    # SAC
    env_id: str = "Hopper-v4"
    total_timesteps: int = 1_000_000
    num_envs: int = 1              # recurrent → single env recommended
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5_000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 1
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    # POMDP helpers
    partial_obs: bool = False
    frame_stack: int = 1

    # LSTM
    use_lstm: bool = True
    lstm_hidden_dim: int = 128
    lstm_layers: int = 1


# ─────────────────────── env factory with wrappers ────────────────────
def make_env(env_id, seed, idx, capture_video, run_name, partial, nstack):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video and idx == 0 else None)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        if partial:
            from cleanrl_extra.wrappers import PartialObsWrapper
            env = PartialObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if nstack > 1:
            env = FrameStack(env, nstack)
            env = FlattenObservation(env)
        env.action_space.seed(seed)
        return env
    return thunk


# ───────────────────────── networks ───────────────────────────────────
class SoftQNetwork(nn.Module):
    def __init__(self, env, use_lstm, hdim, nlayers):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))
        self.use_lstm = use_lstm
        if use_lstm:
            self.enc = RecurrentEncoder(obs_dim, hdim, nlayers)
            in_dim = hdim + act_dim
        else:
            in_dim = obs_dim + act_dim
        self.fc1, self.fc2 = nn.Linear(in_dim, 256), nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.hx = None  # will hold (h,c)

    def forward(self, obs, act):
        if self.use_lstm:
            feat, self.hx = self.enc(obs.unsqueeze(0), self.hx)
        else:
            feat = obs
        x = torch.cat([feat, act], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


LOG_STD_MAX, LOG_STD_MIN = 2, -5


class Actor(nn.Module):
    def __init__(self, env, use_lstm, hdim, nlayers):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        self.use_lstm = use_lstm
        if use_lstm:
            self.enc = RecurrentEncoder(obs_dim, hdim, nlayers)
            in_dim = hdim
        else:
            in_dim = obs_dim
        self.fc1, self.fc2 = nn.Linear(in_dim, 256), nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, env.single_action_space.shape[0])
        self.fc_logstd = nn.Linear(256, env.single_action_space.shape[0])
        self.register_buffer("a_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0))
        self.register_buffer("a_bias",  torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0))
        self.hx = None

    def _dist(self, obs):
        if self.use_lstm:
            feat, self.hx = self.enc(obs.unsqueeze(0), self.hx)
        else:
            feat = obs
        x = F.relu(self.fc1(feat)); x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)

    def get_action(self, obs):
        dist = self._dist(obs)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.a_scale + self.a_bias
        logp = dist.log_prob(x_t) - torch.log(self.a_scale * (1 - y_t.pow(2)) + 1e-6)
        return action, logp.sum(1, keepdim=True), torch.tanh(dist.mean) * self.a_scale + self.a_bias


# ──────────────────────────── main ────────────────────────────────────
if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # logging
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, name=run_name, config=vars(args),
                   monitor_gym=False, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hparams", "\n".join(f"{k}: {v}" for k, v in vars(args).items()))

    # seeding & device
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # env
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video,
                                               run_name, args.partial_obs, args.frame_stack)])
    obs, _ = envs.reset(seed=args.seed)

    # nets
    actor = Actor(envs, args.use_lstm, args.lstm_hidden_dim, args.lstm_layers).to(device)
    qf1 = SoftQNetwork(envs, args.use_lstm, args.lstm_hidden_dim, args.lstm_layers).to(device)
    qf2 = SoftQNetwork(envs, args.use_lstm, args.lstm_hidden_dim, args.lstm_layers).to(device)
    qf1_t, qf2_t = SoftQNetwork(envs, args.use_lstm, args.lstm_hidden_dim, args.lstm_layers).to(device), \
                   SoftQNetwork(envs, args.use_lstm, args.lstm_hidden_dim, args.lstm_layers).to(device)
    qf1_t.load_state_dict(qf1.state_dict()); qf2_t.load_state_dict(qf2.state_dict())

    q_opt = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    a_opt = optim.Adam(actor.parameters(), lr=args.policy_lr)

    # α
    if args.autotune:
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_opt = optim.Adam([log_alpha], lr=args.q_lr)
        target_entropy = -np.prod(envs.single_action_space.shape)
        alpha = log_alpha.exp().item()
    else:
        alpha = args.alpha

    # buffer
    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space,
                      device, n_envs=1, handle_timeout_termination=False)

    start = time.time()
    for gstep in range(args.total_timesteps):
        if gstep < args.learning_starts:
            act_np = np.array([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                a, _, _ = actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device))
            act_np = a.cpu().numpy()

        next_obs, rews, terms, truncs, infos = envs.step(act_np)

        # reset LSTM state when episode ends
        if args.use_lstm and (terms[0] or truncs[0]):
            actor.hx = None; qf1.hx = qf2.hx = None

        if "final_info" in infos and infos["final_info"][0] is not None:
            ep = infos["final_info"][0]["episode"]
            writer.add_scalar("charts/episodic_return", ep["r"], gstep)
            writer.add_scalar("charts/episodic_length", ep["l"], gstep)

        real_next = next_obs.copy()
        if truncs[0]:
            real_next[0] = infos["final_observation"][0]
        rb.add(obs, real_next, act_np, rews, terms, infos)
        obs = next_obs

        # ─────────────── training step ───────────────
        if gstep >= args.learning_starts:
            batch = rb.sample(args.batch_size)

            # target-Q
            with torch.no_grad():
                na, nlogp, _ = actor.get_action(batch.next_observations)
                q1_t = qf1_t(batch.next_observations, na)
                q2_t = qf2_t(batch.next_observations, na)
                min_q_t = torch.min(q1_t, q2_t) - alpha * nlogp
                target_q = batch.rewards.flatten() + (1 - batch.dones.flatten()) * args.gamma * min_q_t.view(-1)

            # critic update
            q1 = qf1(batch.observations, batch.actions).view(-1)
            q2 = qf2(batch.observations, batch.actions).view(-1)
            q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            q_opt.zero_grad(); q_loss.backward(); q_opt.step()

            # actor & α
            if gstep % args.policy_frequency == 0:
                a, logp_a, _ = actor.get_action(batch.observations)
                q1_pi, q2_pi = qf1(batch.observations, a), qf2(batch.observations, a)
                a_loss = (alpha * logp_a - torch.min(q1_pi, q2_pi)).mean()
                a_opt.zero_grad(); a_loss.backward(); a_opt.step()

                # --- α-tuning -----------------
                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(batch.observations)

                    alpha_loss = -(log_alpha.exp() * (log_pi + target_entropy)).mean()

                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()
                    alpha = log_alpha.exp().item()

            # target-net soft-update
            if gstep % args.target_network_frequency == 0:
                with torch.no_grad():
                    for p, tp in zip(qf1.parameters(), qf1_t.parameters()):
                        tp.data.mul_(1 - args.tau).add_(args.tau * p.data)
                    for p, tp in zip(qf2.parameters(), qf2_t.parameters()):
                        tp.data.mul_(1 - args.tau).add_(args.tau * p.data)

            if gstep % 100 == 0:
                writer.add_scalar("losses/critic", q_loss.item(), gstep)
                writer.add_scalar("losses/actor", a_loss.item(), gstep)
                writer.add_scalar("losses/alpha", alpha, gstep)
                writer.add_scalar("charts/SPS", int(gstep / (time.time() - start)), gstep)
                print(f"SPS {int(gstep / (time.time() - start))}")

    envs.close(); writer.close()
