# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import itertools
import os
from pathlib import Path
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from typing import Annotated
import tyro.conf
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers.utils import RunningMeanStd
import shimmy  # this is now necessary
import gymnasium_robotics

gym.register_envs(shimmy)  # unnecessary but prevents IDEs from complaining
gym.register_envs(gymnasium_robotics)

from robot.sim.summary_logger import SummaryLogger
import robot.sim.envs

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    torch_threads: int = 4
    description: str = 'default'
    run_id: str = 'missing_run_id'
    record_results: bool = False
    render: bool = False
    vec_env: bool = False
    target_entropy_coef: float = 1.0

    # continuing setting
    log_freq: int = 1000
    episodic: bool = False
    r_pi_sample: bool = False
    r_pi_update: bool = False
    r_pi_alpha: float = 0.0003
    term_cost: float = None
    term_cost_coef: float = None
    term_delay: float = 0

    # no reset
    no_reset: Annotated[bool, tyro.conf.FlagConversionOff] = False
    terminate_when_unhealthy: bool = True
    log_stats: bool = False
    layer_norm: Annotated[bool, tyro.conf.FlagConversionOff] = False
    ln_elementwise_affine: Annotated[bool, tyro.conf.FlagConversionOff] = False
    reacher_sparse_reward: Annotated[bool, tyro.conf.FlagConversionOff] = True
    reacher_joint_limit: Annotated[bool, tyro.conf.FlagConversionOff] = True
    agent_reset: str = None
    agent_reset_freq: int = 5000
    bias_reset: str = None
    bias_reset_freq: int = 5000
    eps_greedy: bool = False
    fixed_mu: float = 0.0
    fixed_sigma: float = 1.0
    ac_penalty_lambda: float = None
    pnorm: Annotated[bool, tyro.conf.FlagConversionOff] = False
    mean_q: Annotated[bool, tyro.conf.FlagConversionOff] = False
    zero_mean_init: str = None
    ob_ac_norm: Annotated[bool, tyro.conf.FlagConversionOff] = False
    ob_ac_norm_tau1: float = .001
    ob_ac_norm_tau2: float = .001

    # rnd
    rnd: Annotated[bool, tyro.conf.FlagConversionOff] = False
    rnd_coef: float = 1.0
    rnd_update_proportion: float = .25
    rnd_obs_norm_init_steps: int = 50


def make_env(env_id, seed, idx, capture_video, run_name, render_mode=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, layer_norm=False, elementwise_affine=False, pnorm=False):
        super().__init__()
        self.layer_norm = layer_norm
        self.pnorm = pnorm
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        if layer_norm:
            self.ln1 = nn.LayerNorm(256, elementwise_affine=elementwise_affine)
            self.ln2 = nn.LayerNorm(256, elementwise_affine=elementwise_affine)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        if self.layer_norm:
            x = F.relu(self.ln1(self.fc1(x)))
            x = F.relu(self.ln2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        if self.pnorm:
            norm = x.norm(p=2, dim=-1, keepdim=True)
            norm = norm.clamp(min=1e-8) # avoid div by zero
            x = torch.div(x, norm)
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, avg_reward_alpha=.0003, layer_norm=False, elementwise_affine=False, pnorm=False):
        super().__init__()
        self.layer_norm = layer_norm
        self.pnorm = pnorm
        self.r_pi = 0
        # self.r_pi = -.15
        self.r_pi_alpha = avg_reward_alpha
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        if layer_norm:
            self.ln1 = nn.LayerNorm(256, elementwise_affine=elementwise_affine)
            self.ln2 = nn.LayerNorm(256, elementwise_affine=elementwise_affine)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        if not self.layer_norm:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else:
            x = F.relu(self.ln1(self.fc1(x)))
            x = F.relu(self.ln2(self.fc2(x)))
        if self.pnorm:
            norm = x.norm(p=2, dim=-1, keepdim=True)
            norm = norm.clamp(min=1e-8) # avoid div by zero
            x = torch.div(x, norm)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, squash=True, custom_std=None):
        mean, log_std = self(x)
        std = log_std.exp() if custom_std is None else custom_std

        normal = torch.distributions.Normal(mean, std)
        eps = torch.randn(mean.shape)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        x_t = mean + std * eps
        if squash:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
        else:
            action = x_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        if squash:
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        if squash:
            mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, std

    def update_r_pi(self, rewards):
        self.r_pi += self.r_pi_alpha * (rewards - self.r_pi)

class RNDModel(nn.Module):
    def __init__(self, env):
        super().__init__()

        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Target network
        self.target = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature

def set_biases_to_zero(module):
    for m in module.modules():
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0.0)

def weight_init(m):
    if isinstance(m, nn.Linear):
        # print(m)
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)

def initialize_agent(envs, args, device, agent_reset=None):
    actor = Actor(envs, avg_reward_alpha=args.r_pi_alpha, layer_norm=args.layer_norm, elementwise_affine=args.ln_elementwise_affine, pnorm=args.pnorm).to(device)
    qf1 = SoftQNetwork(envs, args.layer_norm, elementwise_affine=args.ln_elementwise_affine, pnorm=args.pnorm).to(device)
    qf2 = SoftQNetwork(envs, args.layer_norm, elementwise_affine=args.ln_elementwise_affine, pnorm=args.pnorm).to(device)
    qf1_target = SoftQNetwork(envs, args.layer_norm, elementwise_affine=args.ln_elementwise_affine, pnorm=args.pnorm).to(device)
    qf2_target = SoftQNetwork(envs, args.layer_norm, elementwise_affine=args.ln_elementwise_affine, pnorm=args.pnorm).to(device)
    actor.apply(weight_init)
    qf1.apply(weight_init)
    qf2.apply(weight_init)
    if args.zero_mean_init is not None:
        if args.zero_mean_init in ['pi', 'both']:
            actor.fc_mean.weight.data.fill_(0.0)
            actor.fc_logstd.weight.data.fill_(0.0)
        if args.zero_mean_init in ['q', 'both']:
            qf1.fc3.weight.data.fill_(0.0)
            qf2.fc3.weight.data.fill_(0.0)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if agent_reset == 'alpha':
        args.autotune = False

    # Automatic entropy tuning
    log_alpha, target_entropy, a_optimizer = None, None, None
    if args.autotune:
        if agent_reset == 'target_entropy':
            args.target_entropy_coef = np.random.uniform(-.675, 1)
            print(f'target entropy coef: {args.target_entropy_coef}')
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item() * args.target_entropy_coef
        print(f'target entropy: {target_entropy}')
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
        if agent_reset == 'alpha':
            alpha = np.random.uniform(.01, 1)
            print(f'alpha: {alpha}')

    return actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, alpha, log_alpha, target_entropy, a_optimizer

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    assert not (args.r_pi_sample and args.r_pi_update)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = None
    if not args.record_results:
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    torch.set_num_threads(args.torch_threads)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    render_mode = 'human' if args.render else None

    if args.vec_env:
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name, render_mode)])
    else:
        env_kwargs = {}
        if 'Cntg' in args.env_id and args.term_cost is not None:
            env_kwargs = {
                'term_cost': args.term_cost,
            }
        if 'HopperCntg' in args.env_id and args.term_cost_coef is not None:
            env_kwargs.update({
                'term_cost_coef': args.term_cost_coef,
            })
        if args.no_reset:
            if 'dm_control' in args.env_id:
                env_kwargs.update({'task_kwargs': {'time_limit': float('inf')}})
            else:
                env_kwargs.update({'max_episode_steps': int(1e9)})
                if args.env_id in [
                    'Hopper-v4', 'HopperCntg-v4', 'Walker2d-v4', 'Walker2dCntg-v4',
                    'Ant-v4', 'AntCntg-v4', 'Humanoid-v4', 'HumanoidCntg-v4',
                ]:
                    env_kwargs.update({'terminate_when_unhealthy': args.terminate_when_unhealthy})
        if 'AntMazeNoReset' in args.env_id:
            env_kwargs.update({'continuing_task': True, 'reset_target': True, 'terminate_when_unhealthy': True, 'healthy_z_range': (.3, 2.0)})
        if 'PointMazeNoReset' in args.env_id:
            env_kwargs.update({'continuing_task': True, 'reset_target': True})
        if 'ReacherNoReset' in args.env_id:
            env_kwargs.update({'sparse_reward': args.reacher_sparse_reward, 'joint_limit': args.reacher_joint_limit})

        envs = gym.make(args.env_id, render_mode=render_mode, **env_kwargs)
        envs = gym.wrappers.FlattenObservation(envs)
        envs = gym.wrappers.RecordEpisodeStatistics(envs)
        envs.action_space.seed(args.seed)
        envs.single_observation_space = envs.observation_space
        envs.single_action_space = envs.action_space
        print(f'obs: {envs.single_observation_space}\nacs: {envs.single_action_space}')

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, alpha, log_alpha, target_entropy, a_optimizer = initialize_agent(envs, args, device, args.agent_reset)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    logs = {'returns': [], 'r_bar': []}
    slog = SummaryLogger(args.log_freq, logs, writer, args.record_results)
    og_return = 0

    if args.log_stats:
        recent_obs_1000, recent_obs_5000 = [], []
        n_hits = 0
        actor.fixed_mu, actor.fixed_sigma = args.fixed_mu, args.fixed_sigma

    if args.ob_ac_norm:
        ob_low, ob_high = np.ones_like(envs.single_observation_space.low) * np.inf, np.ones_like(envs.single_observation_space.high) * -np.inf

    if args.rnd:
        rnd_model = RNDModel(envs).to(device)
        rnd_optimizer = optim.Adam(list(rnd_model.predictor.parameters()), lr=3e-4)
        rnd_rew_rms = RunningMeanStd()
        rnd_obs_rms = RunningMeanStd(shape=envs.single_observation_space.shape)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step % 1000 == 0 and not args.record_results:
            print(f'steps: {global_step}')
        elif global_step % 10000 == 0 and args.record_results:
            print(f'steps: {global_step}')

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            if args.vec_env:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions = envs.single_action_space.sample()
        else:
            actions, _, _, _, _ = actor.get_action(torch.Tensor(obs).to(device))

            actions = actions.detach().cpu().numpy()

            if args.eps_greedy:
                rand_ac_every, rand_ac_for = 100, 5
                if global_step >= 5000 and global_step % rand_ac_every < rand_ac_for:
                    # actions = envs.single_action_space.sample()
                    actions, _, _, _, _ = actor.get_action(torch.Tensor(obs).to(device), custom_std=torch.ones((1,)).to(device))
                    actions = actions.detach().cpu().numpy()
                else:
                    actions, _, _, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                    actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if args.ob_ac_norm:
            ob_low, ob_high = np.minimum(ob_low, obs), np.maximum(ob_high, obs)
            ob_center = (ob_high + ob_low) / 2.0
            ac_center = (envs.single_action_space.high + envs.single_action_space.low) / 2.0
            rewards -= args.ob_ac_norm_tau1 * np.linalg.norm(obs - ob_center) ** 2 \
                     + args.ob_ac_norm_tau2 * np.linalg.norm(actions - ac_center) ** 2

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if args.vec_env:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
        sample_rewards = rewards - actor.r_pi if args.r_pi_sample else rewards
        sample_terms = terminations if args.episodic else False
        # sample_terms = terminations
        if terminations:
            real_next_obs, _ = envs.reset()
        rb.add(obs, real_next_obs, actions, sample_rewards, sample_terms, infos)
        actor.update_r_pi(rewards)

        # logging average reward
        slog.log('r_bar', rewards, global_step)
        if (global_step + 1) % args.log_freq == 0 and not args.record_results:
            print(f"global_step={global_step}, r_bar={logs['r_bar'][-1]['r_bar']}")

        # logging state variance
        if args.log_stats:
            recent_obs_1000.append(obs)
            recent_obs_5000.append(obs)
            if (global_step + 1) % 1000 == 0:
                recent_obs_1000 = np.array(recent_obs_1000)
                slog.log('obs_1000_var', recent_obs_1000.var(0).sum(), global_step)
                recent_obs_1000 = []
            if (global_step + 1) % 5000 == 0:
                recent_obs_5000 = np.array(recent_obs_5000)
                slog.log('obs_5000_var', recent_obs_5000.var(0).sum(), global_step)
                recent_obs_5000 = []

            n_hits += int(rewards > 0) # only valid for reacher sparse reward
            slog.log('n_hits', n_hits, global_step)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = real_next_obs
        if truncations:
            obs, _ = envs.reset()

        # logging original average reward and episodic return
        if 'og_reward' in infos:
            slog.log('og_r_bar', infos['og_reward'], global_step)
            og_return += infos['og_reward']
            if terminations or truncations:
                if 'og_return' not in logs:
                    logs['og_return'] = []
                logs['og_return'].append({'t': global_step, 'og_return': og_return})
                og_return = 0

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if not args.record_results:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                logs['returns'].append({'t': global_step, 'G': float(info['episode']['r'])})
                break
        elif 'episode' in infos:
            if not args.record_results:
                print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                writer.add_scalar('charts/episodic_return', infos['episode']['r'], global_step)
                writer.add_scalar('charts/episodic_length', infos['episode']['l'], global_step)
            logs['returns'].append({'t': global_step, 'G': float(infos['episode']['r'])})

        # agent resets
        if args.agent_reset is not None and (global_step + 1) % args.agent_reset_freq == 0:
            actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, alpha, log_alpha, target_entropy, a_optimizer = initialize_agent(envs, args, device, args.agent_reset)

        # bias resets
        if args.bias_reset is not None and (global_step + 1) % args.bias_reset_freq == 0:
            if args.bias_reset in ['pi', 'both']:
                set_biases_to_zero(actor)
            if args.bias_reset in ['q', 'both']:
                set_biases_to_zero(qf1)
                set_biases_to_zero(qf2)
                set_biases_to_zero(qf1_target)
                set_biases_to_zero(qf2_target)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and rb.size() > args.learning_starts:
            data = rb.sample(args.batch_size)

            if args.rnd:
                if global_step <= args.learning_starts + args.rnd_obs_norm_init_steps:
                    # ppo-rnd updates the obs_rms for 50 steps before training
                    rnd_obs_rms.update(np.array(data.observations))
                else:
                    rnd_obs = (data.observations - torch.from_numpy(rnd_obs_rms.mean).to(device)) / torch.sqrt(torch.from_numpy(rnd_obs_rms.var).to(device) + 1e-8)
                    rnd_obs = rnd_obs.float()
                    target_feature = rnd_model.target(rnd_obs)
                    predict_feature = rnd_model.predictor(rnd_obs)
                    rnd_rews = ((target_feature - predict_feature).pow(2).sum(1) / 2).data
                    rnd_rew_rms.update(np.array(rnd_rews)) # cleanrl's ppo-rnd updates the rew_rms before scaling
                    rnd_rews_scaled = rnd_rews / np.sqrt(rnd_rew_rms.var + 1e-8)
                    rnd_obs_rms.update(np.array(data.observations))

                    # rnd update
                    rnd_next_obs = (data.next_observations - torch.from_numpy(rnd_obs_rms.mean).to(device)) / torch.sqrt(torch.from_numpy(rnd_obs_rms.var).to(device) + 1e-8)
                    rnd_next_obs = rnd_next_obs.float()
                    predict_next_feature, target_next_feature = rnd_model(rnd_next_obs)
                    rnd_loss = F.mse_loss(
                        predict_next_feature, target_next_feature.detach(), reduction="none"
                    ).mean(-1)

                    mask = torch.rand(len(rnd_loss), device=device)
                    mask = (mask < args.rnd_update_proportion).type(torch.FloatTensor).to(device)
                    rnd_loss = (rnd_loss * mask).sum() / torch.max(
                        mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                    )
                    rnd_optimizer.zero_grad()
                    rnd_loss.backward()
                    rnd_optimizer.step()

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                if args.mean_q:
                    stacked_qf_next_target = torch.stack((qf1_next_target, qf2_next_target), dim=0)
                    min_qf_next_target = torch.mean(stacked_qf_next_target, dim=0) - alpha * next_state_log_pi
                else:
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                if args.r_pi_update:
                    diff_r = data.rewards.flatten() - actor.r_pi
                    if args.rnd and global_step > args.learning_starts + args.rnd_obs_norm_init_steps:
                        diff_r += args.rnd_coef * rnd_rews_scaled
                    next_q_value = diff_r + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                    if not args.record_results:
                        writer.add_scalar("misc./diff. r", (data.rewards.flatten() - actor.r_pi).mean(), global_step)
                        if args.rnd and global_step > args.learning_starts + args.rnd_obs_norm_init_steps:
                            writer.add_scalar("misc./diff. r rnd", (data.rewards.flatten() - actor.r_pi + rnd_rews_scaled).mean(), global_step)
                else:
                    rs = data.rewards.flatten()
                    if args.rnd and global_step > args.learning_starts + args.rnd_obs_norm_init_steps:
                        rs += args.rnd_coef * rnd_rews_scaled
                    next_q_value = rs + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, mean, std = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    if args.ac_penalty_lambda is not None: # penalty (RL variance)
                        actor_loss = ((alpha * log_pi) - min_qf_pi + args.ac_penalty_lambda * pi.norm(p=2, dim=1, keepdim=True)).mean()
                    else:
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                        # actor_loss = (- min_qf_pi).mean()
                        # actor_loss = (alpha * log_pi) .mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, mean, std, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()

                        alpha = log_alpha.exp().item()
                        # if (log_pi + target_entropy).mean() < 0:
                        #     alpha *= -1
                    else:
                        pass

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0 and not args.record_results:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("misc./actor_r_pi", actor.r_pi, global_step)
                writer.add_scalar("charts/-log_pi", -log_pi.mean(), global_step)
                writer.add_scalar("charts/(-log_pi).std()", (-log_pi).std(), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)


    if args.record_results:
        log_dir = Path(__file__).parent / Path(f'exp/data/{args.description}/{args.seed}')
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f'{args.run_id}.json', 'wt') as f:
            import json
            json.dump({'logs': logs}, f, indent=4)

    envs.close()
    if not args.record_results:
        writer.close()

