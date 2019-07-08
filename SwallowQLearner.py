# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:19:05 2019

@author: Sergio
"""
import torch
import numpy as np
from libs.perceptron import SLP

# EPISILON_MIN: vamos aprendiendo mientras el incremento de aprendizaje supera dicho valor
# MAX_NUM_EPISODIES: número máximo de iteraciones que estamos dispuestos a realizar
# STEPS_PER_EPISODE: Número máximo de pasos a realizar por episodio
# ALPHA: ratio aprendizaje del agente
# GAMMA: factor de descuento del agente
# NUM_DISCRETE_BINS: número de divisiones en el caso de discretizar el espacio continuo.
MAX_NUM_EPISODIES = 50000
STEPS_PER_EPISODE = 200
EPSILON_NIM = 0.005
max_num_steps = MAX_NUM_EPISODIES * STEPS_PER_EPISODE
EPSION_DECAY = 500 * EPSILON_NIM / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

class SwallowQLearner(object):
    def  __init__(self,environment):
        self.obs_shape = environment.observation_space.shape
        self.obs_high = environment.observation_space.high
        self.obs_low = environment.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high - self.obs_low)/self.obs_bins
        
        self.action_shape = environment.action_space.n
        self.Q = SLP(self.obs_shape,self.action_shape)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(),lr = 1e-5)
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0  
        
    def discretize(self, obs):
        return tuple(((obs-self.obs_low)/self.bin_width).astype(int))
        
    def get_action(self, obs):
        discrete_obs = self.discretize(obs)
        # Seleccion de la accion en base a Epsion-Greedy
        if self.epsilon > EPSILON_NIM:
            self.epsilon -= EPSION_DECAY
        if np.random.random() > self.epsilon: # Con probabilidad 1-epsilon, elegimos la mejor posible
            return np.argmax(self.Q(discrete_obs).data.to(torch.device('cpu')).numpy())
        else:
            return np.random.choice([a for a in range(self.action_shape)]) # Con probabilidad epsilon, elegimos una al azar
        
    def learn(self, obs, action, reward, net_obs):
        td_target = reward + self.gamma * torch.max(self.q(net_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action],td_target)
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()