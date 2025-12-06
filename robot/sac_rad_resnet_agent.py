import copy
import torch
import time
import queue

import relod.utils as utils
import numpy as np
from torch import nn
from torch import optim
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from relod.algo.sac_rad_buffer import AsyncRadReplayBuffer, RadReplayBuffer
from relod.algo.rl_agent import BaseLearner, BasePerformer
from relod.algo.models import ActorModel, CriticModel
from relod.algo.sac_rad_agent import SACRADPerformer, SACRADLearner

from robot.models import ActorModelResNet18, CriticModelResNet18, EncoderClassifierModelResNet18
from robot.sim.scale import RunningScale

class SACRADResNetPerformer(SACRADPerformer):
    def __init__(self, args) -> None:
        self._args = args
        self._args.device = torch.device(args.device)

        if not 'conv' in self._args.net_params:  # no image
            self._args.image_shape = (0, 0, 0)

        # self._actor = ActorModel(
        self._actor = ActorModelResNet18(
                                 self._args.image_shape,
                                 self._args.proprioception_shape,
                                 self._args.action_shape[0],
                                 self._args.net_params,
                                 self._args.rad_offset,
                                 self._args.layernorm,
                                 self._args.pnorm).to(self._args.device)

        # self._critic = CriticModel(
        self._critic = CriticModelResNet18(
                                 self._args.image_shape,
                                 self._args.proprioception_shape,
                                 self._args.action_shape[0],
                                 self._args.net_params,
                                 self._args.rad_offset,
                                 self._args.layernorm,
                                 self._args.pnorm).to(self._args.device)
        self._critic_target = copy.deepcopy(self._critic) # also copies the encoder instance
        self._actor.encoder = self._critic.encoder

        if isinstance(self._actor.encoder, EncoderClassifierModelResNet18):
            self.reset_batchnorm_stats(self._actor.encoder.encoder_head)
            self.reset_batchnorm_stats(self._actor.encoder.encoder_fc)

            # self.reset_batchnorm_stats(self._actor.encoder.backbone)
            # self.reset_batchnorm_stats(self._actor.classifier_head)
            # self.reset_batchnorm_stats(self._actor.classifier_fc)

        self.train(is_training=args.mode != 'e')

    def reset_batchnorm_stats(self, module):
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()

    def train(self, is_training=True):
        # This function is used to temporarily set the network to evaluation mode every time an action is sampled.
        if isinstance(self._actor.encoder, EncoderClassifierModelResNet18):
            self._actor.trunk.train(is_training)
            self._critic.Q1.train(is_training)
            self._critic.Q2.train(is_training)
            self._critic_target.Q1.train(is_training)
            self._critic_target.Q2.train(is_training)
            self._actor.encoder.encoder_head.train(is_training)
            self._actor.encoder.encoder_fc.train(is_training)
            self._critic_target.encoder.backbone.eval()
            self._critic_target.encoder.encoder_head.eval()
            self._critic_target.encoder.encoder_fc.eval()

            self._actor.encoder.backbone.eval()
            # self._actor.encoder.backbone.train(is_training)

            self._actor.classifier_head.eval()
            self._actor.classifier_fc.eval()
            # self._actor.classifier_head.train(True)
            # self._actor.classifier_fc.train(True)

            self._actor.encoder.backbone.requires_grad_(False)
            self._actor.encoder.encoder_head.requires_grad_(True)
            self._actor.encoder.encoder_fc.requires_grad_(True)
            self._actor.classifier_head.requires_grad_(False)
            self._actor.classifier_fc.requires_grad_(False)
            self._critic_target.encoder.backbone.requires_grad_(False)
            self._critic_target.encoder.encoder_head.requires_grad_(False)
            self._critic_target.encoder.encoder_fc.requires_grad_(False)

            # print(f'critic target params: {list(self._critic_target.encoder.backbone.named_parameters())}')
            # print(f'critic target params: {(self._critic_target.encoder.backbone.layer1[0].bn1.state_dict())}')
            # print(f'critic backbone: {self._critic.encoder.backbone.training}')
            # print(f'critic encoder_head: {self._critic.encoder.encoder_head.training}')
            # print(f'critic encoder_fc: {self._critic.encoder.encoder_fc.training}')
            # print(f'critic target backbone: {self._critic_target.encoder.backbone.training}')
            # print(f'critic target encoder_head: {self._critic_target.encoder.encoder_head.training}')
            # print(f'critic target encoder_fc: {self._critic_target.encoder.encoder_fc.training}')
        else:
            self._actor.train(is_training)
            self._critic.train(is_training)
            self._critic_target.train(is_training)
        self.is_training = is_training

    def sample_action(self, ob):
        # sample action for data collection
        with utils.eval_mode(self):
            (image, propri) = ob

            with torch.inference_mode():
                if image is not None:
                    image = torch.FloatTensor(image).to(self._args.device)
                    image.unsqueeze_(0)

                if propri is not None:
                    propri = torch.FloatTensor(propri).to(self._args.device)
                    propri.unsqueeze_(0)

                classifier_logits = None
                if isinstance(self._actor.encoder, EncoderClassifierModelResNet18):
                    mu, pi, _, log_std, unsquashed_mu, unsquashed_pi, classifier_logits = self._actor(
                        image, propri, random_rad=False, compute_pi=True, compute_log_pi=False,
                    )
                else:
                    mu, pi, _, log_std, unsquashed_mu, unsquashed_pi = self._actor(
                        image, propri, random_rad=False, compute_pi=True, compute_log_pi=False,
                    )
                # print('mu:', mu.cpu().data.numpy().flatten())
                # print('std:', log_std.exp().cpu().data.numpy().flatten())
                action = pi.cpu().data.numpy().flatten()

        return action, {'unsquashed_mu': unsquashed_mu, 'unsquashed_pi': unsquashed_pi, 'mu': mu, 'pi': pi, 'log_std': log_std, 'classifier_logits': classifier_logits}

class TargetEntropyAdapter:
    def __init__(self, T, learning_rate):
        self.T = T
        self.lr = learning_rate
        self.tec_min, self.tec_max = -.68, 1
        self.target_entropy_coef = torch.ones(1) * self.tec_min
        self.use_high_alpha = torch.zeros(1)
        self.rewards = []
        self.prev_r_bar = 0

    def update(self, step, rewards):
        self.rewards.append(rewards)
        if step % self.T == 0:
            r_bar = np.mean(np.array(self.rewards))
            if r_bar > self.prev_r_bar:
                self.target_entropy_coef += self.lr * (self.tec_max - self.tec_min) * 1 # decrease entropy
                self.use_high_alpha.fill_(0.0)
            else:
                self.target_entropy_coef -= self.lr * (self.tec_max - self.tec_min) * 1 # increase entropy
                self.use_high_alpha.fill_(1.0)
            self.target_entropy_coef.clip_(self.tec_min, self.tec_max)
            self.prev_r_bar = r_bar
            self.rewards = []

class SACRADResNetLearner(SACRADLearner):
    def __init__(self, args, performer=None) -> None:
        self.t0 = time.perf_counter()
        self.avg_reward = torch.zeros(1).to(args.device)
        self.avg_reward.requires_grad_ = False
        self.avg_reward.share_memory_()
        self.avg_reward_alpha = .0003
        self.r_pi_update = args.r_pi_update
        self.te_adapter = None
        if args.adapt_te:
            self.te_adapter = TargetEntropyAdapter(100, 1)
            self.te_adapter.target_entropy_coef.requires_grad_ = False
            self.te_adapter.target_entropy_coef.share_memory_()
            self.te_adapter.use_high_alpha.requires_grad_ = False
            self.te_adapter.use_high_alpha.share_memory_()
        self.q_scaler = None
        if args.scale_q:
            self.q_scaler = RunningScale(device=args.device)
            # self.q_scaler.share_memory_()
        self.learned_target_entropy = -np.prod(args.action_shape) * -.67
        self.prev_Q1 = None
        self.td_err = None
        self.delta_Q1 = None
        if args.load_model > -1:
            self.avg_reward *= float(np.loadtxt(args.model_dir + '/avg_reward.txt'))
            print(f'loaded avg. reward: {self.avg_reward}')
        super().__init__(args, performer)

    def _update_critic(self, images, proprioceptions, actions, rewards, next_images, next_proprioceptions, dones):
        with torch.inference_mode():
            if isinstance(self._actor.encoder, EncoderClassifierModelResNet18):
                _, policy_actions, log_pis, _, _, _, _ = self._actor(next_images, next_proprioceptions)
            else:
                _, policy_actions, log_pis, _, _, _ = self._actor(next_images, next_proprioceptions)
            target_Q1, target_Q2 = self._critic_target(next_images, next_proprioceptions, policy_actions)
            target_V = torch.min(target_Q1, target_Q2) - self._alpha.detach() * log_pis
            if self._args.bootstrap_terminal:
                # enable infinite bootstrap
                target_Q = rewards - (int(self.r_pi_update) * self.avg_reward) + (self._args.discount * target_V)
            else:
                target_Q = rewards - (int(self.r_pi_update) * self.avg_reward) + ((1.0 - dones) * self._args.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self._critic(images, proprioceptions, actions, detach_encoder=False)

        critic_loss = torch.mean((current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2)

        # Optimize the critic
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self._critic_optimizer.step()

        # calculate the average TD error
        if self._args.learn_target_entropy:
            with torch.inference_mode():
                if self.prev_Q1 is not None:
                    bootstrap_coef = 1.0 if self._args.bootstrap_terminal else 1. - dones
                    # FIXME: The next line for TD error calculation is wrong.
                    # self.td_err = rewards + bootstrap_coef * self._args.discount * current_Q1 - self._alpha.detach() * log_pis - self.prev_Q1
                    next_Q1, _ = self._critic(next_images, next_proprioceptions, policy_actions, detach_encoder=True)
                    self.td_err = rewards + bootstrap_coef * self._args.discount * (next_Q1 - self._alpha.detach() * log_pis) - current_Q1
                    # self.td_err = rewards + bootstrap_coef * self._args.discount * next_Q1 - current_Q1
                    self.delta_Q1 = current_Q1.mean() - self.prev_Q1.mean()
                self.prev_Q1 = current_Q1

        critic_stats = {
            'train_critic/loss': critic_loss.item(),
            'train_critic/q1_values': current_Q1.mean().item(),
            'train_critic/-alpha_logpi': (-self._alpha.detach() * log_pis).mean().item(),
            'train_critic/target_V': target_V.mean().item(),
        }

        if self._args.learn_target_entropy and self.prev_Q1 is not None and self.td_err is not None:
            critic_stats.update({
                'train_critic/prev_Q1': self.prev_Q1.mean().item(),
                'train_critic/td_err': self.td_err.mean().item(),
                'train_critic/delta_Q1': self.delta_Q1.item(),
                'train_critic/delta_Q1_sign': torch.sign(self.delta_Q1).item(),
            })

        return critic_stats

    def _update_actor_and_alpha(self, images, proprioceptions):
        # NOTE one can shave off around 10ms from the update by doing forward
        # pass through the resnet encoder once for both actor and critic

        # detach encoder, so we don't update it with the actor loss
        if isinstance(self._actor.encoder, EncoderClassifierModelResNet18):
            mus, pis, log_pis, log_stds, _, _, _ = self._actor(images, proprioceptions ,detach_encoder=True)
        else:
            mus, pis, log_pis, log_stds, _, _ = self._actor(images, proprioceptions ,detach_encoder=True)
        actor_Q1, actor_Q2 = self._critic(images, proprioceptions, pis, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        if self.q_scaler is not None:
            self.q_scaler.update(actor_Q)
            actor_Q = self.q_scaler(actor_Q)
            log_pis = log_pis.div(pis.shape[-1])
        if self.te_adapter is not None and self._args.adapt_te_high_alpha is not None:
            my_alpha = self._alpha.detach()
            if self.te_adapter.use_high_alpha.gt(0.5).item():
                my_alpha = self._args.adapt_te_high_alpha
            actor_loss = (my_alpha * log_pis - actor_Q).mean()
        else:
            actor_loss = (self._alpha.detach() * log_pis - actor_Q).mean()
        # actor_loss = (self._alpha.detach() * log_pis - actor_Q + torch.pow(unsquashed_mus, 2).mean(axis=1)).mean()

        entropy = 0.5 * log_stds.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_stds.sum(dim=-1)

        # optimize the actor
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        if self._args.learn_target_entropy and self.td_err is not None:
            # self.learned_target_entropy += -1. * torch.sign(self.td_err.mean())
            self.learned_target_entropy += -1. * self.td_err.mean()
            # self.learned_target_entropy += -.01 * torch.sign(self.delta_Q1)
            # self.learned_target_entropy += -1. * self.delta_Q1
            self.learned_target_entropy = self.learned_target_entropy.clip(self._target_entropy, -.67 * self._target_entropy)
            alpha_loss = (self._alpha * (-log_pis - self.learned_target_entropy).detach()).mean()
        elif self._args.anneal_target_entropy:
            my_te = self._target_entropy
            end_n_updates = 1500000
            if self._num_updates < end_n_updates:
                te_max, te_min = -.67 * self._target_entropy, self._target_entropy
                my_te = te_max - (self._num_updates / end_n_updates) * (te_max - te_min)
            alpha_loss = (self._alpha * (-log_pis - self._target_entropy).detach()).mean()
        elif self.te_adapter is not None:
            with torch.no_grad():
                _, _, log_pis, _, _, _ = self._actor(images, proprioceptions, detach_encoder=True)
                my_te = self._target_entropy * self.te_adapter.target_entropy_coef.item()
            alpha_loss = (self._alpha * (-log_pis - my_te).detach()).mean()
        else:
            alpha_loss = (self._alpha * (-log_pis - self._target_entropy).detach()).mean()

        if not (self.te_adapter is not None and self._args.adapt_te_high_alpha is not None and self.te_adapter.use_high_alpha.gt(0.5).item()):
            self._log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._log_alpha_optimizer.step()

        actor_stats = {
            'train_actor/loss': actor_loss.item(),
            'train_actor/target_entropy': self._target_entropy.item(),
            'train_actor/learned_target_entropy': self.learned_target_entropy.item(),
            'train_actor/entropy': -log_pis.mean().item(),
            'train_alpha/loss': alpha_loss.item(),
            'train_alpha/value': self._alpha.item(),
            'train/entropy': entropy.mean().item(),
        }
        if self.te_adapter is not None:
            actor_stats.update({
                'train_actor/target_entropy_coef': self.te_adapter.target_entropy_coef.item(),
                'train_actor/te_adapter target_entropy': my_te,
            })
        return actor_stats

    def soft_update_params(self, net, target_net, tau):
        # Copying the state dict like below also copies the statistics of the batchnorm layers
        # TODO What should happen to batchnorm layer's num_batches_tracked?
        net_sd, target_net_sd = net.state_dict(), target_net.state_dict()
        for k in net_sd.keys():
            target_net_sd[k].copy_(tau * net_sd[k].data + (1 - tau) * target_net_sd[k].data)

    def _soft_update_target(self):
        self.soft_update_params(
            self._critic.Q1, self._critic_target.Q1, self._args.critic_tau
        )
        self.soft_update_params(
            self._critic.Q2, self._critic_target.Q2, self._args.critic_tau
        )
        self.soft_update_params(
            self._critic.encoder, self._critic_target.encoder,
            self._args.encoder_tau
        )

    def _update(self, images, propris, actions, rewards, next_images, next_propris, dones):
        tic = time.time()
        # regular update of SAC_RAD, sequentially augment data and train
        if images is not None:
            images = torch.as_tensor(images, device=self._args.device).float()
            next_images = torch.as_tensor(next_images, device=self._args.device).float()
        if propris is not None:
            propris = torch.as_tensor(propris, device=self._args.device).float()
            next_propris = torch.as_tensor(next_propris, device=self._args.device).float()
        actions = torch.as_tensor(actions, device=self._args.device)
        rewards = torch.as_tensor(rewards, device=self._args.device)
        dones = torch.as_tensor(dones, device=self._args.device)

        stats = self._update_critic(images, propris, actions, rewards, next_images, next_propris, dones)
        if self._num_updates % self._args.actor_update_freq == 0:
            actor_stats = self._update_actor_and_alpha(images, propris)
            stats = {**stats, **actor_stats}
        if self._num_updates % self._args.critic_target_update_freq == 0:
            self._soft_update_target()
        stats['train/batch_reward'] = rewards.mean().item()
        stats['train/num_updates'] = self._num_updates
        self._num_updates += 1
        if self._num_updates % 100 == 0:
            print("Update {} took {:.4f}s to update the model".format(self._num_updates, time.time()-tic))

        actor = self._actor

        return stats

    def _init_optimizers(self):
        self._actor_optimizer = optim.Adam(
            self._actor.parameters(), lr=self._args.actor_lr, betas=(0.9, 0.999)
        )

        self._critic_optimizer = optim.Adam(
            self._critic.parameters(), lr=self._args.critic_lr, betas=(0.9, 0.999)
        )

        self._log_alpha_optimizer = optim.Adam(
            [self._log_alpha], lr=self._args.alpha_lr, betas=(0.9, 0.999)
        )

    def push_sample(self, ob, action, reward, next_ob, done):
        (image, propri) = ob
        (next_image, next_propri) = next_ob

        if self._args.async_mode:
            try:
                self._sample_queue.put_nowait((image, propri, action, reward, next_image, next_propri, done))
            except queue.Full:
                pass
        else:
            self._replay_buffer.add(image, propri, action, reward, next_image, next_propri, done)

    def clear_replay_buffer(self):
        self._sample_queue.put('clear')
        self.clear_minibatch_queue.value = 1

    def _async_update(self):
        # TODO clearing the replay buffer and the minibatch queue should be simpler than this
        # right now this goes into infinite loop if init_steps is too small
        while True:
            # print('cmq: ', self.clear_minibatch_queue.value)
            if self.clear_minibatch_queue.value == 1:
                self.minibatch_queue_lock.acquire()
                try:
                    while self._minibatch_queue.qsize() > 0:
                        print(f'mbq clearing...: {self._minibatch_queue.qsize()}')
                        self._minibatch_queue.get()
                except queue.Empty:
                    pass
                print(f'mbq cleared: {self._minibatch_queue.qsize()}')
                self.clear_minibatch_queue.value = 0
                self.minibatch_queue_lock.release()
            else:
                try:
                    # print(f'mbq size: {self._minibatch_queue.qsize()}')
                    self._update_queue.put_nowait(self._update(*self._minibatch_queue.get()))
                except queue.Full:
                    pass

    def save_policy_to_file(self, model_dir, step):
        super().save_policy_to_file(model_dir, step)
        np.savetxt(model_dir + '/avg_reward.txt', np.array([self.avg_reward.item()]))

    def get_avg_reward(self):
        return self.avg_reward

    def update_avg_reward(self, step, reward):
        self.avg_reward += self.avg_reward_alpha * (reward - self.avg_reward)
        if self.te_adapter is not None:
            self.te_adapter.update(step, reward)

