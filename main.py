# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:35:32 2020

@author: Ali
"""

from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from dqn_agent import Agent


# please do not modify the line below
env = UnityEnvironment(file_name="../envionments/Banana_Windows_x86_64/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


# create an agent
my_agent = Agent(state_size, action_size, seed=0)


env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state

score_record = []
score_window = deque(maxlen=100)
score = 0
i_episode = 0

for i in range(100000):
    state = env_info.vector_observations[0]             # get the current state
    action = my_agent.act(state)                        # get an action using epsilon greedy policy
    env_info = env.step(action)[brain_name]             # send the action to the environment
    next_state = env_info.vector_observations[0]        # get the next state
    reward = env_info.rewards[0]                        # get the reward
    done = env_info.local_done[0]                       # see if episode has finished
    
    my_agent.step(state, action, reward, next_state, done)
    
    score += reward

    
    if done:
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        score_window.append(score)
        score_record.append(np.mean(score_window))
        score = 0
        i_episode +=1
        
        if i_episode%100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score_record[-1]))
            
        if i_episode>100:
            if np.all(np.array(score_window)>13):
                print('Congradulations, you have passed the criteria')
                break


plt.plot(score_record)


my_agent.save_model()

env.close()