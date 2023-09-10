import logging
import os
import copy
import torch
import torch.nn.functional as F
import numpy as np

class ARSAC:
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

        self.target_entropy = -action_num
        # self.target_entropy = -torch.prod(torch.Tensor([action_num]).to(self.device)).item()

        init_temperature = 0.01
        """self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=self.hyper_params["LR_ENTROPY"]
        )"""
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)

        # new ones
        self.alpha = 0.7 # 0.4 0.6
        self.min_priority = 1
        self.noise_clip = 0.5  # self.noise_clip = noise_clip
        self.policy_noise = 0.2  # self.policy_noise = policy_noise

        # RD-PER parameters
        self.scale_r = 1.0
        self.scale_s = 1.0
        self.update_step = 0
        """
        sac_exp_hyper_params = {
            "ACTOR_SIZE": [400, 300],
            "CRITIC_SIZE": [400, 300],
            "GAMMA": 0.98,
            "TAU": 0.02,
            "LR_ACTOR": 7.3e-4,
            "LR_QF1": 7.3e-4,
            "LR_QF2": 7.3e-4,
            "LR_ENTROPY": 7.3e-4,
            "BUFFER_SIZE": 1000000,
            "BATCH_SIZE": 256,
            "AUTO_ENTROPY_TUNING": True,
            "INITIAL_RANDOM_ACTION": 10000,
            "TOTAL_STEPS": 2000000,
            "MULTIPLE_LEARN": 64,
            "TRAIN_FREQ": 64,
            "PER_ALPHA": 0.7,
            "PER_BETA": 0.4,
            "PER_EPS": 1e-6,
            "WEIGHT_DECAY": 0.0
        }
        """


    def div(self,target):
        return target[:, 0], target[:, 1], target[:, 2:]

    def select_action_from_policy(self, state, evaluation=False):
        # note that when evaluating this algorithm we need to select mu as action so _, _, action = self.actor_net.sample(state_tensor)
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            if evaluation is False:
                action, _, _, = self.actor_net.sample(state_tensor)
            else:
                _, _, action, = self.actor_net.sample(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    @property
    def alpha_entropy(self):
        return self.log_alpha.exp()

    # def train_policy(self, experiences):
    def train_policy(self, replay_buffer, batch_size):

        self.learn_counter += 1
        info = {}

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
            next_actions, next_log_pi, _ = self.actor_net.sample(next_states)
            target_q_values_one, target_q_values_two = self.target_critic_net(next_states, next_actions)
            next_values1, _, _ = self.div(target_q_values_one)
            next_values2, _, _ = self.div(target_q_values_two)
            min_next_target = torch.minimum(next_values1, next_values2).reshape(-1, 1)  # torch.min
            #min_qf_next_target = min_qf_next_target - self.log_alpha.exp().detach() * next_state_log_pi
            target_q_values = min_next_target - self.alpha_entropy * next_log_pi
            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        diff_td1 = F.mse_loss(values1.reshape(-1, 1), q_target, reduction='none')
        diff_td2 = F.mse_loss(values2.reshape(-1, 1), q_target, reduction='none')
        critic_total_loss1 = (diff_td1 + self.scale_r * diff_rew1 + self.scale_s * diff_next_states1)
        critic_total_loss2 = (diff_td2 + self.scale_r * diff_rew2 + self.scale_s * diff_next_states2)

        critic1_loss = (diff_rew1.reshape(-1, 1))
        critic2_loss = (diff_rew2.reshape(-1, 1))
        #critic_loss_total = self.huber(critic_loss_1) + self.huber(critic_loss_2)

        # print(f'loss_td: {critic4_loss}')
        # print(f'loss_rew: {critic2_loss}')

        critic_loss_total = (critic_total_loss1 * weights + critic_total_loss2 * weights)
        # critic_loss_total = critic1_loss  + critic2_loss
        # train critic
        self.critic_net.optimiser.zero_grad()
        torch.mean(critic_loss_total).backward()
        self.critic_net.optimiser.step()

        pi, log_pi, _ = self.actor_net.sample(states)
        qf1__pi, qf2__pi = self.critic_net(states, pi)
        qf1_pi, _, _ = self.div(qf1__pi)
        qf2_pi, _, _ = self.div(qf2__pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)
        actor_loss = torch.mean(((self.alpha_entropy * log_pi) - min_qf_pi) * weights)

        # Update the Actor
        self.actor_net.optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net.optimiser.step()

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

        # way 2 of calculate priority

        priority = torch.max(critic1_loss, critic2_loss).clamp(min=self.min_priority).pow(
            self.alpha).cpu().data.numpy().flatten()
        replay_buffer.update_priority(indices, priority)



        # update the temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


        if self.learn_counter % self.policy_update_freq == 0:
            for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))



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