import os
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F

class ARTD3:
    def __init__(self,
                 actor_network,
                 critic_network,
                 gamma,
                 tau,
                 action_num,
                 state_dim,
                 device):

        self.type = "policy"
        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.target_actor_net = copy.deepcopy(self.actor_net).to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.gamma = gamma
        self.tau = tau

        self.learn_counter = 0  # self.total_it = 0
        self.policy_update_freq = 2  # policy_freq=2

        self.action_num = action_num
        self.state_dim = state_dim
        self.device = device

        # new ones
        self.alpha = 0.7 # 0.4 0.6
        self.min_priority = 1
        self.noise_clip = 0.5  # self.noise_clip = noise_clip
        self.policy_noise = 0.2  # self.policy_noise = policy_noise

        # MAPER parameters
        self.scale_r = 1.0
        self.scale_s = 1.0
        self.update_step = 0

    def div(self,target):
        return target[:, 0], target[:, 1], target[:, 2:]

    def select_action_from_policy(self, state, evaluation=False, noise_scale=0.1):
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # todo check if really need this line
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()

            if not evaluation:
                # this is part the TD3 too, add noise to the action
                noise = np.random.normal(0, scale=noise_scale, size=self.action_num)
                action = action + noise
                action = np.clip(action, -1, 1)
        self.actor_net.train()
        return action

    # def train_policy(self, experiences):
    def train_policy(self, replay_buffer, batch_size):

        self.learn_counter += 1

        # Sample replay buffer
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)

        #print(f'State: {dones.shape}')
        dones = dones.reshape(-1, 1)

        # Get current Q estimates way2 (2)
        q_values_one, q_values_two = self.critic_net(states.detach(), actions.detach())
        values1, rew1, next_states1 = self.div(q_values_one)
        values2, rew2, next_states2 = self.div(q_values_two)


        # diff_rew1 = F.smooth_l1_loss(rew1.reshape(-1, 1), rewards)
        diff_rew1 = 0.5 * torch.pow(rew1.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0).reshape(-1, 1)
        diff_rew2 = 0.5 * torch.pow(rew2.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0).reshape(-1, 1)
        diff_next_states1 = 0.5 * torch.mean(torch.pow(next_states1.reshape(-1, self.state_dim) - next_states.reshape(-1, self.state_dim), 2.0), -1).reshape(-1, 1)
        diff_next_states2 = 0.5 * torch.mean(torch.pow(next_states2.reshape(-1, self.state_dim) - next_states.reshape(-1, self.state_dim), 2.0), -1).reshape(-1, 1)

        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)
            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.target_critic_net(next_states, next_actions)
            next_values1, _, _ = self.div(target_q_values_one)
            next_values2, _, _ = self.div(target_q_values_two)
            target_q_values = torch.min(next_values1, next_values2).reshape(-1,1)  # torch.min

            #rew = (rew1.reshape(-1, 1) + rew2.reshape(-1, 1)) / 2
            #rewards = rew.reshape(-1, 1)
            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        #############################################
        # way1

        # this is need to change based on huber function
        # critic loss

        diff_td1 = F.mse_loss(values1.reshape(-1, 1), q_target, reduction='none')
        diff_td2 = F.mse_loss(values2.reshape(-1, 1), q_target, reduction='none')
        critic3_loss = (diff_td1 + self.scale_r * diff_rew1 + self.scale_s * diff_next_states1)
        critic4_loss = (diff_td2 + self.scale_r * diff_rew2 + self.scale_s * diff_next_states2)

        critic1_loss = (diff_rew1.reshape(-1, 1))
        critic2_loss = (diff_rew2.reshape(-1, 1))

       # print(f'loss_td: {critic4_loss}')
        #print(f'loss_rew: {critic2_loss}')

        critic_loss_total = (critic3_loss * weights + critic4_loss * weights)
        # critic_loss_total = critic1_loss  + critic2_loss
        # train critic
        self.critic_net.optimiser.zero_grad()
        torch.mean(critic_loss_total).backward()
        self.critic_net.optimiser.step()

        # policy loss

        numpy_td = torch.cat([diff_td1, diff_td2], -1)
        numpy_td = torch.mean(numpy_td, 1)
        numpy_td = numpy_td.view(-1, 1)
        numpy_td = numpy_td[:, 0].detach().data.cpu().numpy()

        numpy_r = torch.cat([diff_rew1, diff_rew2], -1)
        numpy_r = torch.mean(numpy_r, 1)
        numpy_r = numpy_r.view(-1, 1)
        numpy_r = numpy_r[:, 0].detach().data.cpu().numpy()

        numpy_s = torch.cat([diff_next_states1, diff_next_states2], -1)
        numpy_s = torch.mean(numpy_s, 1)
        numpy_s = numpy_s.view(-1, 1)
        numpy_s = numpy_s[:, 0].detach().data.cpu().numpy()
        numpy_next_q_value = q_target[:, 0].detach().data.cpu().numpy()



        """
        self.mean_td += np.mean(numpy_td)
        self.mean_q += np.mean(numpy_next_q_value)
        self.mean_r += np.mean(numpy_r)
        self.mean_s += np.mean(numpy_s)
        """
        ############################
        # way 1 of calculate priority
        """
        new_priorities = np.array(
            [numpy_next_q_value,
             numpy_td + self.scale_s * numpy_s + self.scale_r * numpy_r])
        new_indices = np.array(indices)
        """
        """
        new_priorities = (numpy_td + self.scale_s * numpy_s + self.scale_r * numpy_r)
        replay_buffer.update_priority(indices, new_priorities)
        """

        ############################
        # way 2 of calculate priority

        priority = torch.max(critic1_loss, critic2_loss).clamp(min=self.min_priority).pow(self.alpha).cpu().data.numpy().flatten()
        replay_buffer.update_priority(indices, priority)

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            # (1)
            actor_q_one, actor_q_two = self.critic_net(states.detach(), self.actor_net(states.detach()))
            actor_q_values = torch.minimum(actor_q_one, actor_q_two)
            actor_val, _, _ = self.div(actor_q_values)
            ###############
            # way1
            #actor_loss = -(weights * actor_val).mean()
            ###############
            # way2
            actor_loss = -actor_val.mean()

            # Optimize the actor
            self.actor_net.optimiser.zero_grad()
            actor_loss.backward()
            self.actor_net.optimiser.step()

            # Update target network params
            for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        ################################################
        # Update Scales
        if self.update_step == 0:
            self.scale_r = np.mean(numpy_td) / (np.mean(numpy_r))
            self.scale_s = np.mean(numpy_td) / (np.mean(numpy_s))
        self.update_step += 1

    def huber(self, x):
        return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()

    # If min_priority=1, this can be simplified.
    """def PAL(self, x):
        return torch.where(
                x.abs() < self.min_priority,
                (self.min_priority ** self.alpha) * 0.5 * x.pow(2),
                self.min_priority * x.abs().pow(1. + self.alpha) / (1. + self.alpha)
            ).mean()"""

    def save_models(self, filename):
        dir_exists = os.path.exists("models")

        if not dir_exists:
            os.makedirs("models")
        torch.save(self.actor_net.state_dict(), f'models/{filename}_actor.pht')
        torch.save(self.critic_net.state_dict(), f'models/{filename}_critic.pht')
        logging.info("models has been loaded...")

    def load_models(self, filename):
        self.actor_net.load_state_dict(torch.load(f'models/{filename}_actor.pht'))
        self.critic_net.load_state_dict(torch.load(f'models/{filename}_critic.pht'))
        logging.info("models has been loaded...")