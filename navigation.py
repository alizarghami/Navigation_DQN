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


class Navigation(object):
    def __init__(self, env_path, criteria=13, seed=0, prioritize_er=False, double_dqn=False, drop_out = False):
        """
        Creates a Navigtion instance

        Parameters
        ----------
        env_path : str
            Path to the unity environment.
        criteria : int, optional
            The score we aim to reach. The default is 13.
        seed : int, optional
            Seed to use. The default is 0.
        """
        self.env = UnityEnvironment(file_name = env_path)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]       
        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.brain.vector_observation_space_size
        self.agent = Agent(self.state_size, self.action_size, seed=seed, prioritize_er=prioritize_er, double_dqn=double_dqn, drop_out=drop_out)
        self.criteria = criteria
        
        self.score_record = []
        self.score_window = deque(maxlen=100)
        
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
    
    
    def reset_env(self):
       self.env_info = self.env.reset(train_mode=True)[self.brain_name] 


    def run_episode(self, mode=0, eps=0.):
        """
        Runs one full episode

        Parameters
        ----------
        mode : int, optional
            Whether the episod is for training or evaluation(0:Evaluation, 1:Training). The default is 0.
        eps : float, optional
            Epsiolon value used in epsilon-greedy policy (Only matters in training mode). The default is 0..

        Returns
        -------
        score : int
            The total achived score in an episode.

        """
        if mode==0:
            eps = 0.
        done = False
        score = 0 
        
        while not done:
            state = self.env_info.vector_observations[0]             # get the current state
            action = self.agent.act(state, eps=eps)                  # get an action using epsilon greedy policy
            self.env_info = self.env.step(action)[self.brain_name]   # send the action to the environment
            next_state = self.env_info.vector_observations[0]        # get the next state
            reward = self.env_info.rewards[0]                        # get the reward
            done = self.env_info.local_done[0]                       # see if episode has finished
            
            if mode == 1:
                self.agent.step(state, action, reward, next_state, done)
            
            score += reward
            
        self.reset_env()                                             # reset the environment
            
        return score
    
    
    def run_evaluation_episode(self):
        score = self.run_episode(mode=0)
        return score
    
    def evaluate(self, runs=100):
        """
        Generates episodes to evaluate the current model

        Parameters
        ----------
        runs : int, optional
            Number of episodes to use in evaluation. The default is 100.
        """
        score_record = []
        
        print('Evaluation in progress...')
        for i in range(runs):
            score = self.run_evaluation_episode()
            score_record.append(score)
            
        ave_score = np.mean(score_record)
        
        print('System evaluated with an average score of {} in {} runs'.format(ave_score, runs))
        
        
    def run_training_episode(self, eps):
        score = self.run_episode(mode=1, eps=eps)
        return score
    
    
    def train(self, max_episodes= 1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """
        Generates episodes and trains the agent until the score criteria is met

        Parameters
        ----------
        max_episodes : int, optional
            Maximum episodes to generate. The default is 1000.
        eps_start : float, optional
            Maximum epsilon value for epsilon-greedy policy. The default is 1.0.
        eps_end : float, optional
            Minimum epsilon value for epsilon-greedy policy. The default is 0.01.
        eps_decay : float, optional
            The rate which epsilon decays in epsilon-greedy policy. The default is 0.995.
            
        Returns
        -------
        success (bool): Whether the criteria was reached during the training or not
        """
        success = False
        i_episode = 0
        eps = eps_start
        
        print('Training in progress...')
        for i in range(max_episodes):
            score = self.run_training_episode(eps=eps)
                        
            self.score_window.append(score)
            self.score_record.append(np.mean(self.score_window))
            
            i_episode += 1
            eps = max(eps_end, eps_decay*eps) # decrease epsilon

            if i_episode%100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, self.score_record[-1]))
                
            if i_episode>100:
                if np.mean(self.score_window)>self.criteria:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, self.score_record[-1]))
                    success = True
                    break

        if success:
            print('Criteria reached after {} episodes'.format(i_episode))
        else:
            print('Failed to reach Criteria after {} episodes'.format(i_episode))

        self.plot_training_progress()
        return success


    def plot_training_progress(self):
        """Plots the recorded scores achieved in the training phase"""
        if self.score_record:
            plt.plot(self.score_record)
            plt.ylabel('Average score')
            plt.xlabel('Episode')
            plt.title('Average score for last 100 episodes in the training phase')
        else:
            print('No progress made yet...')


    def reset_records(self):
        """Resets all the recorded scores"""
        self.score_record = []
        self.score_window = deque(maxlen=100)
        
    
    def reset_model(self):
        self.agent.reset_models()


    def save_model(self, file_name=None):
        """
        saves the current model
        
            Parameters:
                file_name (str): Path to the save location
        """
        try:
            if file_name:
                self.agent.save_model(file_name)
            else:
                self.agent.save_model()
            print('Model saved successfully')
            return 1
        except:
            print('Failed to save model')
            return 0
            
            
    def load_model(self, file_name=None):
        """
        Loads a pre-trained model
        
            Parameters:
                file_name (str): Path to the saved model
        """
        try:
            if file_name:
                self.agent.load_model(file_name)
            else:
                self.agent.load_model()
            print('Model loaded successfully')
            return 1
        except:
            print('Failed to load model')
            return 0


    def close_env(self):
        """Closes the unity environment"""
        self.env.close()
        