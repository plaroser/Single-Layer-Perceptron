# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:19:05 2019

@author: Sergio
"""
import torch
import numpy as np
from libs.perceptron import SLP
from utils.decay_shecule import LinearDecayShedule
import random

# EPISILON_MIN: vamos aprendiendo mientras el incremento de aprendizaje supera dicho valor
# MAX_NUM_EPISODIES: número máximo de iteraciones que estamos dispuestos a realizar
# STEPS_PER_EPISODE: Número máximo de pasos a realizar por episodio
# ALPHA: ratio aprendizaje del agente
# GAMMA: factor de descuento del agente
# NUM_DISCRETE_BINS: número de divisiones en el caso de discretizar el espacio continuo.
MAX_NUM_EPISODIES = 100000
STEPS_PER_EPISODE = 300

class SwallowQLearner(object):
    def  __init__(self,environment, learning_rate = 0.005, gamma = 0.98):
        self.obs_shape = environment.observation_space.shape
        
        self.action_shape = environment.action_space.n
        self.Q = SLP(self.obs_shape,self.action_shape)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(),lr = learning_rate)
        self.gamma = gamma
        
        self.epsilon_max = 1.0  
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecayShedule(initial_value = self.epsilon_max,
                                                fnal_value = self.epsilon_min,
                                                max_steps = 0.5 * MAX_NUM_EPISODIES * STEPS_PER_EPISODE)
        self.step_num = 0
        self.policy = self.epsilon_greddy_Q
        
    def get_action(self, obs):
        return self.policy(obs)
    
    def epsilon_greddy_Q(self, obs):
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device('cpu')).numpy())
        return action
        
    def learn(self, obs, action, reward, net_obs):
        td_target = reward + self.gamma * torch.max(self.q(net_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action],td_target)
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()