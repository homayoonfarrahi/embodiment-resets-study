import signal
import sys
import torch
import argparse
import relod.utils as utils
import time
import numpy as np
import cv2
import os
from pathlib import Path

from relod.logger import Logger
from relod.algo.comm import MODE
from relod.algo.local_wrapper import LocalWrapper
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.envs.visual_ur5_reacher.configs.ur5_config import config
from relod.envs.visual_ur5_min_time_reacher.env import VisualReacherMinTimeEnv, MonitorTarget
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from robot.bouncy_ball.bouncy_ball_env import BouncyBallEnv

from robot.sac_rad_resnet_agent import SACRADResNetLearner, SACRADResNetPerformer
from robot.sac_rad_resnet_buffer import AsyncRadResNetReplayBuffer, RadResNetReplayBuffer

config = {
    
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    
    'latent': 50,

    'mlp': [
        [-1, 1024], # first hidden layer
        [1024, 1024], 
        [1024, -1] # output layer
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Local remote visual UR5 Reacher')
    # environment
    parser.add_argument('--setup', default='Visual-UR5-min-time')
    parser.add_argument('--env', default='ur5', type=str)
    parser.add_argument('--ur5_ip', default='129.128.159.210', type=str)
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--image_height', default=90, type=int)
    parser.add_argument('--target_type', default='size', type=str)
    parser.add_argument('--random_action_repeat', default=1, type=int)
    parser.add_argument('--agent_action_repeat', default=1, type=int)
    parser.add_argument('--image_history', default=3, type=int)
    parser.add_argument('--joint_history', default=1, type=int)
    parser.add_argument('--ignore_joint', default=False, action='store_true')
    parser.add_argument('--episode_length_time', default=10.0, type=float)
    parser.add_argument('--dt', default=0.04, type=float)
    parser.add_argument('--size_tol', default=0.015, type=float)
    parser.add_argument('--center_tol', default=0.1, type=float)
    parser.add_argument('--reward_tol', default=2.0, type=float)
    parser.add_argument('--reset_penalty_steps', default=70, type=int)
    parser.add_argument('--reward', default=-1, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=200000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=2000, type=int) 
    parser.add_argument('--env_steps', default=300000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=False, action='store_true')
    parser.add_argument('--max_updates_per_step', default=0.6, type=float)
    parser.add_argument('--update_every', default=50, type=int)
    parser.add_argument('--update_epochs', default=50, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    parser.add_argument('--bootstrap_terminal', default=0, type=int)
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-3, type=float)
    # agent
    parser.add_argument('--remote_ip', default='localhost', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='e', type=str, help="Modes in ['r', 'l', 'rl', 'e'] ")
    # misc
    parser.add_argument('--run_type', default='experiment', type=str)
    parser.add_argument('--description', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--work_dir', default='results/', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--plot_learning_curve', default=False, action='store_true')
    parser.add_argument('--xtick', default=1200, type=int)
    parser.add_argument('--show_cam_feed', default=True, action='store_true')
    parser.add_argument('--save_image', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=100000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')

    parser.add_argument('--learn_clsf', default=False, action='store_true')
    parser.add_argument('--store_actions', default=False, action='store_true')
    parser.add_argument('--anneal_target_entropy', default=False, action='store_true')
    parser.add_argument('--learn_target_entropy', default=False, action='store_true')
    parser.add_argument('--episodic', default=False, action='store_true')
    parser.add_argument('--r_pi_update', default=False, action='store_true')
    parser.add_argument('--r_pi_sample', default=False, action='store_true')
    parser.add_argument('--continual', default=False, action='store_true')
    parser.add_argument('--layernorm', default=False, action='store_true')
    parser.add_argument('--pnorm', default=False, action='store_true')
    parser.add_argument('--adapt_te', default=False, action='store_true')
    parser.add_argument('--adapt_te_high_alpha', default=None, type=float)
    parser.add_argument('--scale_q', default=False, action='store_true')

    args = parser.parse_args()
    assert args.mode in ['r', 'l', 'rl', 'e']
    assert args.reward < 0 and args.reset_penalty_steps >= 0
    args.async_mode = not args.sync_mode
    return args

def main():
    args = parse_args()

    if args.mode == 'r':
        mode = MODE.REMOTE_ONLY
    elif args.mode == 'l':
        mode = MODE.LOCAL_ONLY
    elif args.mode == 'rl':
        mode = MODE.REMOTE_LOCAL
    elif args.mode == 'e':
        mode = MODE.EVALUATION
    else:
        raise  NotImplementedError()

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.work_dir += f'/{args.env}/{args.description}/timeout={args.episode_length_time:.0f}/seed={args.seed}'
    args.model_dir = args.work_dir+'/models'
    args.return_dir = args.work_dir+'/returns'
    if mode != MODE.EVALUATION:
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.return_dir, exist_ok=True)
    if mode == MODE.LOCAL_ONLY:
        L = Logger(args.return_dir, use_tb=args.save_tb)
        run_name = f'{args.description}__{args.env}__{Path(__file__).stem[:-len(".py")]}__{args.seed}'
        writer = SummaryWriter(f'runs/{run_name}')

    if args.save_image:
        args.image_dir = args.work_dir+'/images'
        if mode == MODE.LOCAL_ONLY or mode == MODE.EVALUATION:
            os.makedirs(args.image_dir, exist_ok=True)

    env = BouncyBallEnv(
        setup = args.setup,
        ip = args.ur5_ip,
        seed = args.seed,
        camera_id = args.camera_id,
        image_width = args.image_width,
        image_height = args.image_height,
        target_type = args.target_type,
        image_history = args.image_history,
        joint_history = args.joint_history,
        episode_length = args.episode_length_time,
        dt = args.dt,
        size_tol = args.size_tol,
        center_tol = args.center_tol,
        reward_tol = args.reward_tol,
        continual = args.continual,
    )

    utils.set_seed_everywhere(args.seed, None)
    # # mt = MonitorTargetGravityBall()
    # # mt = MonitorTargetGravityBallProcess()
    # mt = MonitorTargetGravityBallPyGameProcess()
    # mt.start()
    # mt.reset_plot()
    # input('go?')
    cv2.namedWindow('raw')
    cv2.moveWindow('raw', 2000, 200)
    image, prop = env.reset()
    image_to_show = np.transpose(image, [1, 2, 0])
    image_to_show = image_to_show[:,:,-3:]
    feed_size_coef = 10 if mode == MODE.EVALUATION else 1
    feed_size = tuple(int(dim * feed_size_coef) for dim in reversed(image_to_show.shape[:2]))
    image_to_show = cv2.resize(image_to_show, feed_size)
    cv2.imshow('raw', image_to_show)
    cv2.waitKey(1)
    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    args.net_params = config
    # args.replay_buffer_class = AsyncRadResNetReplayBuffer if args.async_mode else RadResNetReplayBuffer

    episode_length_step = int(args.episode_length_time / args.dt)
    agent = LocalWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    agent.init_performer(SACRADResNetPerformer, args)
    agent.init_learner(SACRADResNetLearner, args, agent.performer)

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)

    # if args.load_model > -1:
    #     agent.load_policy_from_file(args.model_dir, args.load_model)
    
    # First inference took a while (~1 min), do it before the agent-env interaction loop
    if mode != MODE.REMOTE_ONLY:
        agent.performer.sample_action((image, prop))
        agent.performer.sample_action((image, prop))
        agent.performer.sample_action((image, prop))

    # Experiment block starts
    experiment_done = False
    total_steps = 0
    epi_done = 0
    returns = []
    epi_lens = []
    if args.load_model > -1 and mode != MODE.EVALUATION:
        total_steps = args.load_model
        ret_data = np.loadtxt(args.model_dir + '/return.txt')
        epi_lens, returns = ret_data[0], ret_data[1]
        # only load epi_lens and returns up to the load_model checkpoint
        idx = np.cumsum(epi_lens) <= args.load_model
        epi_lens = list(epi_lens[idx])
        returns = list(returns[idx])
        print(f'starting from total_steps: {total_steps}\t and {len(epi_lens)} episodes')

    start_time = time.time()
    print(f'Experiment starts at: {start_time}')

    # start a new episode
    env.total_steps = total_steps
    image, prop = env.reset() 
    agent.send_init_ob((image, prop))
    ret, epi_steps, epi_done = 0, 0, 0
    epi_start_time = time.time()

    while not experiment_done:
        update_cam_feed(args.show_cam_feed, image, feed_size)

        action, ac_info = agent.sample_action((image, prop))
        env.total_steps = total_steps
        next_image, next_prop, reward, epi_done, _ = env.step(action)

        if mode != MODE.EVALUATION:
            sample_done = epi_done if args.episodic else False
            sample_reward = reward - agent._learner.get_avg_reward() if args.r_pi_sample else reward
            agent.push_sample((image, prop), action, sample_reward, (next_image, next_prop), sample_done)
            agent._learner.update_avg_reward(total_steps, reward)
            if args.adapt_te:
                writer.add_scalar('train_actor/target_entropy_coef_mainproc', agent._learner.te_adapter.target_entropy_coef.item(), total_steps)
            stat = agent.update_policy(total_steps)
            if stat is not None and mode == MODE.LOCAL_ONLY:
                for k, v in stat.items():
                    L.log(k, v, total_steps)
                    writer.add_scalar(k, v, total_steps)

        image, prop = next_image, next_prop

        # Log
        total_steps += 1
        ret += reward
        epi_steps += 1
        experiment_done = total_steps >= args.env_steps

        if args.save_model and total_steps % args.save_model_freq == 0 and mode != MODE.EVALUATION:
            agent._learner.pause_update()
            agent.save_policy_to_file(args.model_dir, total_steps)
            # agent.save_buffer()
            utils.save_returns(args.model_dir + '/return.txt', returns, epi_lens)
            agent._learner.resume_update()

        if epi_done: # episode done, save result
            returns.append(ret)
            epi_lens.append(epi_steps)
            if mode == MODE.EVALUATION:
                print(f'episode: {len(returns)}\t return: {ret}')

            if mode != MODE.EVALUATION:
                utils.save_returns(args.return_dir+'/return.txt', returns, epi_lens)

            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - epi_start_time, total_steps)
                L.log('train/episode_reward', ret, total_steps)
                L.log('train/episode', len(returns), total_steps)
                L.dump(total_steps)
                if args.plot_learning_curve:
                    utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)
                writer.add_scalar('train/return', ret, total_steps)
                writer.add_scalar('train/episode', len(returns), total_steps)
                writer.add_scalar('train/avg_reward', agent._learner.get_avg_reward(), total_steps)

            # start a new episode
            image, prop = env.reset() 

            # if (not args.episodic) and mode != MODE.EVALUATION:
            #     # updates are done in parallel
            #     # no need to call update_policy in this case
            #     action, ac_info = agent.sample_action((next_image, next_prop))
            #     rr = env.get_reset_reward()
            #     sample_rr = rr - agent._learner.get_avg_reward() if args.r_pi_sample else rr
            #     agent.push_sample((next_image, next_prop), action, sample_rr, (image, prop), False)
            #     agent._learner.update_avg_reward(rr)

            agent.send_init_ob((image, prop))
            ret, epi_steps, epi_done = 0, 0, 0
            epi_start_time = time.time()

    duration = time.time() - start_time
    agent.save_policy_to_file(args.model_dir, total_steps)

    # Clean up
    env.reset()
    agent.close()
    env.close()

    # always show a learning curve at the end
    if mode == MODE.LOCAL_ONLY:
        utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)
    print(f"Finished in {duration}s")

def update_cam_feed(show_cam_feed, image, feed_size):
    if show_cam_feed:
        image_to_show = np.transpose(image, [1, 2, 0])
        image_to_show = image_to_show[:,:,-3:]
        image_to_show = cv2.resize(image_to_show, feed_size)
        cv2.imshow('raw', image_to_show)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
