
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (Actor, Critic)
from memory import SequentialMemory
from episodic import EpisodicMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

criterion = nn.MSELoss()

class Agent(object):
    def __init__(self, nb_states, nb_actions, args):
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
 
        # Create Actor and Critic Network
        self.actor = Actor(self.nb_states, self.nb_actions, args.init_w)
        self.actor_target = Actor(self.nb_states, self.nb_actions, args.init_w)

        self.critic = Critic(self.nb_states, self.nb_actions, args.init_w)
        self.critic_target = Critic(self.nb_states, self.nb_actions, args.init_w)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.trajectory_length = args.trajectory_length
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        return action

    def select_action(self, state, noise_enable=True, decay_epsilon=True):
        action, _ = self.actor(to_tensor(np.array([state])))
        action = to_numpy(action).squeeze(0)
        if noise_enable == True:
            action += self.is_training * max(self.epsilon, 0)*self.random_process.sample()

        action = np.clip(action, -1., 1.)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        return action

    def reset_lstm_hidden_state(self, done=True):
        self.actor.reset_lstm_hidden_state(done)

    def reset(self):
        self.random_process.reset_states()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
