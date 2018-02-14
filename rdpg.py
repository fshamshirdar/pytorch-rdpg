import numpy as np
import argparse
from copy import deepcopy
import torch
from torch.optim import Adam
import torch.nn.functional as F
import gym

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from memory import EpisodicMemory
from agent import Agent
from util import *

class RDPG(object):
    def __init__(self, env, nb_states, nb_actions, args):
        if args.seed > 0:
            self.seed(args.seed)

        self.env = env

        self.nb_states = nb_states
        self.nb_actions= nb_actions

        self.agent = Agent(nb_states, nb_actions, args)
        self.memory = EpisodicMemory(capacity=args.rmsize, max_episode_length=args.trajectory_length, window_length=args.window_length)
        self.evaluate = Evaluator(args.validate_episodes, args.validate_steps, max_episode_length=args.max_episode_length)

        self.critic_optim  = Adam(self.agent.critic.parameters(), lr=args.rate)
        self.actor_optim  = Adam(self.agent.actor.parameters(), lr=args.prate)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.trajectory_length = args.trajectory_length
        self.max_episode_length = args.max_episode_length
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
        self.warmup = args.warmup
        self.validate_steps = args.validate_steps

        # 
        self.epsilon = 1.0
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()
 
    def train(self, num_iterations, checkpoint_path, debug):
        self.agent.is_training = True
        step = episode = episode_steps = trajectory_steps = 0
        episode_reward = 0.
        state0 = None
        while step < num_iterations:
            episode_steps = 0
            while episode_steps < self.max_episode_length:
                # reset if it is the start of episode
                if state0 is None:
                    state0 = deepcopy(self.env.reset())
                    self.agent.reset()

                # agent pick action ...
                if step <= self.warmup:
                    action = self.agent.random_action()
                else:
                    action = self.agent.select_action(state0)

                # env response with next_observation, reward, terminate_info
                state, reward, done, info = self.env.step(action)
                state = deepcopy(state)

                self.env.render()

                # agent observe and update policy
                self.memory.append(state0, action, reward, done)

                # update 
                step += 1
                episode_steps += 1
                trajectory_steps += 1
                episode_reward += reward
                state0 = deepcopy(state)

                if trajectory_steps >= self.trajectory_length:
                    self.agent.reset_lstm_hidden_state(done=False)
                    trajectory_steps = 0
                    if step > self.warmup:
                        self.update_policy()

                # [optional] save intermideate model
                if step % int(num_iterations/3) == 0:
                    self.agent.save_model(checkpoint_path)

                if done: # end of episode
                    if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

                    # reset
                    state0 = None
                    episode_reward = 0.
                    episode += 1
                    self.agent.reset_lstm_hidden_state(done=True)
                    break

            # [optional] evaluate
            if self.evaluate is not None and self.validate_steps > 0 and step % self.validate_steps == 0:
                policy = lambda x: self.agent.select_action(x, decay_epsilon=False)
                validate_reward = self.evaluate(self.env, policy, debug=False, visualize=False)
                if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

#            if step >= args.warmup and episode > args.bsize:
#                # Update weights
#                agent.update_policy()


    def update_policy(self):
        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        if len(experiences) == 0: # not enough samples
            return

        policy_loss_total = 0
        value_loss_total = 0
        for t in range(len(experiences) - 1): # iterate over episodes
            target_cx = Variable(torch.zeros(self.batch_size, 50)).type(FLOAT)
            target_hx = Variable(torch.zeros(self.batch_size, 50)).type(FLOAT)

            cx = Variable(torch.zeros(self.batch_size, 50)).type(FLOAT)
            hx = Variable(torch.zeros(self.batch_size, 50)).type(FLOAT)

            # we first get the data out of the sampled experience
            state0 = np.stack((trajectory.state0 for trajectory in experiences[t]))
            # action = np.expand_dims(np.stack((trajectory.action for trajectory in experiences[t])), axis=1)
            action = np.stack((trajectory.action for trajectory in experiences[t]))
            reward = np.expand_dims(np.stack((trajectory.reward for trajectory in experiences[t])), axis=1)
            # reward = np.stack((trajectory.reward for trajectory in experiences[t]))
            state1 = np.stack((trajectory.state0 for trajectory in experiences[t+1]))

            target_action, (target_hx, target_cx) = self.agent.actor_target(to_tensor(state1, volatile=True), (target_hx, target_cx))
            next_q_value = self.agent.critic_target([
                to_tensor(state1, volatile=True),
                target_action
            ])
            next_q_value.volatile=False

            target_q = to_tensor(reward) + self.discount*next_q_value

            # Critic update
            current_q = self.agent.critic([ to_tensor(state0), to_tensor(action) ])

            # value_loss = criterion(q_batch, target_q_batch)
            value_loss = F.smooth_l1_loss(current_q, target_q)
            value_loss /= len(experiences) # divide by trajectory length
            value_loss_total += value_loss

            # Actor update
            action, (hx, cx) = self.agent.actor(to_tensor(state0), (hx, cx))
            policy_loss = -self.agent.critic([
                to_tensor(state0),
                action
            ])
            policy_loss /= len(experiences) # divide by trajectory length
            policy_loss_total += policy_loss.mean()

            # update per trajectory
            self.agent.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

            self.agent.actor.zero_grad()
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optim.step()


        # update only once
#        policy_loss_total /= self.batch_size # divide by number of trajectories
#        value_loss_total /= self.batch_size # divide by number of trajectories
#
#        self.agent.critic.zero_grad()
#        value_loss_total.backward()
#        self.critic_optim.step()
#
#        self.agent.actor.zero_grad()
#        policy_loss_total.backward()
#        self.actor_optim.step()

        # Target update
        soft_update(self.agent.actor_target, self.agent.actor, self.tau)
        soft_update(self.agent.critic_target, self.agent.critic, self.tau)

    def test(self, num_episodes, model_path, visualize=True, debug=False):
        if self.agent.load_weights(model_path) == False:
            prRed("model path not found")
            return

        self.agent.is_training = False
        self.agent.eval()
        policy = lambda x: self.agent.select_action(x, noise_enable=False, decay_epsilon=False)

        for i in range(num_episodes):
            validate_reward = self.evaluate(self.env, policy, debug=debug, visualize=visualize, save=False)
            if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
