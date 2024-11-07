# -*- coding: utf-8 -*-

# I need to check detach() for all algo like actor ant this one

import logging
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy
import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
import pydantic
from networks.TD3 import Actor, Critic, PrioritizedReplayBuffer
from algorithms import TD3
from config import LearningConfig

logging.basicConfig(level=logging.INFO)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(env,seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.action_space.seed(seed)

def plot_reward_curve(data_reward):
    data = pd.DataFrame.from_dict(data_reward)
    data.plot(x='step', y='episode_reward', title="Reward Curve")
    plt.show()

def denormalize(action, max_action_value, min_action_value):
    # return action in env range [max_action_value, min_action_value]
    max_range_value = max_action_value
    min_range_value = min_action_value
    max_value_in = 1
    min_value_in = -1
    action_denorm = (action - min_value_in) * (max_range_value - min_range_value) / (
                max_value_in - min_value_in) + min_range_value
    return action_denorm

def normalize(action, max_action_value, min_action_value):
    # return action in algorithm range [-1, +1]
    max_range_value = 1
    min_range_value = -1
    max_value_in = max_action_value
    min_value_in = min_action_value
    action_norm = (action - min_value_in) * (max_range_value - min_range_value) / (
                max_value_in - min_value_in) + min_range_value
    return action_norm

def evaluate_policy_network(env, agent, learning_config):
    evaluation_seed = learning_config.seed
    max_steps_evaluation = 10000
    if max_steps_evaluation == 0:
        return

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    env = gym.make(env.spec.id, render_mode="human")
    state, _ = env.reset(seed=evaluation_seed)

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1
        action = agent.select_action_from_policy(state, evaluation=True)
        action_env = denormalize(action, max_action_value, min_action_value)

        state, reward, done, truncated, _ = env.step(action_env)
        episode_reward += reward

        if done or truncated:
            logging.info(
                f" Evaluation Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

def train(agent,env, memory, max_action_value, min_action_value, learning_config):
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    state, _ = env.reset(seed=learning_config.seed)

    historical_reward = {"step": [], "episode_reward": []}
    logging.info(state)
    for total_step_counter in range(int(learning_config.max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < learning_config.max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{learning_config.max_steps_exploration}")
            action_env = env.action_space.sample()  # action range the env uses [e.g. -2 , 2 for pendulum]
            action = normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]

        else:
            action = agent.select_action_from_policy(state)  # algorithm range [-1, 1]
            action_env = denormalize(action, max_action_value,
                                     min_action_value)  # mapping to env range [e.g. -2 , 2 for pendulum]

        next_state, reward, done, truncated, _ = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state
        episode_reward += reward

        if total_step_counter >= learning_config.max_steps_exploration:
            for _ in range(learning_config.G):
                # experiences = memory.sample(BATCH_SIZE)
                agent.train_policy(memory, learning_config.batch_size)

        if done or truncated:
            logging.info(
                f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Specify the file name for the .csv file
            file_name = "TD3_RD-PER-Walker-v4_seed10_careh_1m.csv"

            rl_data = historical_reward

            # Open the .csv file in write mode
            with open(file_name, mode='w', newline='') as file:
                # Create a CSV writer object
                writer = csv.writer(file)

                # Write the header row
                writer.writerow(["Step", "episode_reward"])

                # Write each data row
                for step, episode_reward in zip(rl_data["step"], rl_data["episode_reward"]):
                    writer.writerow([step + 1, episode_reward])


            #print(f"Data saved to {file_name}.")

            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    plot_reward_curve(historical_reward)

def main():
    learning_config = pydantic.parse_file_as(path="config/learning_config.json", type_=LearningConfig)

    start_time = time.time()
    env = gym.make(
        'HalfCheetah-v4')  # Mujoco: 'HalfCheetah-v4', Humanoid-v4, Swimmer-v4, Ant-v4, InvertedPendulum-v4, Walker2d-v4, "Hopper-v4"
    # Pendulum-v1, BipedalWalker-v3
    observation_size = env.observation_space.shape[0]
    action_num = env.action_space.shape[0]

    max_action_value = env.action_space.high[0]
    min_action_value = env.action_space.low[0]

    set_seed(env, learning_config.seed)
    memory = PrioritizedReplayBuffer(observation_size, action_num)
    actor = Actor(observation_size, action_num, learning_config.actor_lr)
    critic = Critic(observation_size, action_num, learning_config.critic_lr)

    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        gamma=learning_config.gamma,
        tau=learning_config.tau,
        action_num=action_num,
        state_dim= observation_size,
        device=DEVICE,
    )

    train(agent,env, memory, max_action_value, min_action_value, learning_config)
    #evaluate_policy_network(env, agent, learning_config)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time, "seconds")

if __name__ == '__main__':
    main()

