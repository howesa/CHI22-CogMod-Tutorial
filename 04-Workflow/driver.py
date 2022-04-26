# Car driver

import random
import math
import numpy as np
from operator import itemgetter

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

import matplotlib.pyplot as plt

class driver(Env):
    def __init__(self, speed = 17, obs_prob = 0.9, action_noise = 0.02, steer_noise = 0.01, oob_reward = -1, continuous = False):
        self.speed = speed
        # pos x is always between [0,1]. Threshold tells how far from
        # either limit there is negative reward.
        self.threshold = 0.2
        # Into how many bins position belief is put.
        self.n_xb = 50
        # Actions are discrete between [0,max_actions], and translated
        # into radians according to this.
        # (0 = -max_steer, max_actions = max_steer)
        self.max_steer = 0.08

        self.step_time = 0.15

        self.action_noise = action_noise
        self.steer_noise = steer_noise

        self.obs_prob = obs_prob

        self.oob_reward = oob_reward

        self.actions = 10

        self.max_time = 1e10

        self.continuous = continuous
        if self.continuous:
            self.action_space = Box(low=-1,high=1, shape = (1,))
        else:
            self.action_space = Discrete(self.actions)
        self.observation_space = Box(low=0, high=1, shape = (2,))

        self.t_hat = None

        self.log = []

        self.agent = None

        self.training = False
        # When training, use this to close / open eyes
        self.attention_switch_prob = 0.1

        self.reset()

    def reset(self):
        self.has_attention = True
        self.x = 0.5 # in the middle of the lane
        self.time = 0
        # Position belief consists of bins. Add one binlen to the last
        # to include the max x pos.
        #binlen = 1/self.x_bins
        #self.n_xb = len(np.round(np.arange(0,1+binlen, x_binlen),np.ceil(self.x_bins/10)))

        # Set belief
        self.x_b = np.zeros(self.n_xb) + (1 - self.obs_prob) / self.n_xb
        self.x_b[int(self.x * self.n_xb)] = self.obs_prob

        return self.get_belief()

    def get_belief(self):
        entropy = - np.sum(self.x_b * np.log(self.x_b + 1e-5)) / np.log(self.n_xb)
        x = random.choice(np.where(self.x_b == np.max(self.x_b))[0])/self.n_xb
        return [x,entropy]

    def update_belief(self, action):
        # Observe?
        if self.has_attention:
            self.x_b = np.zeros(self.n_xb) + ((1 - self.obs_prob) / self.n_xb)
            if random.random() < self.obs_prob:
                x = self.x
            else:
                x = random.random()

            self.x_b[int(x * self.n_xb)] = self.obs_prob

        # Bayesian belief update using model
        if self.continuous:
            action = self.discretise_action(action)
        a = self.t_hat[action] #[int(self.x*self.n_xb)]

        # Jointly
        b = a * self.x_b[:, np.newaxis]

        self.x_b = b.sum(axis=0)

    def plot_belief(self):
        plt.close()
        x = np.linspace(0, 0.999, self.n_xb)
        plt.scatter(x, self.x_b)
        plt.show()

    # Given a discrete action, return a true steering position
    def action_to_steer(self, action):
        if self.continuous:
            return action*self.max_steer
        else:
            a = self.action_space.n/2
            return (action-a)*self.max_steer/a

    # Scale a [-1,1] continuous action into [0,n.actions[ discrete
    # action.
    def discretise_action(self, action):
        return int((action+1)*(self.actions/2))

    def discrete_action_to_cont(self, action):
        return action/(self.actions/2)-1

    def learn_t_hat(self, x_samples = 200, samples = 1000):
        self.has_attention = True
        self.reset()

        t_hat = np.zeros(shape = (self.actions, self.n_xb, self.n_xb))

        # Sample consequences of actions to x, given a sweep through
        # the space of x.
        for a in range(self.actions):
            for x in np.linspace(0, 0.999, x_samples):
                for i in range(samples):
                    self.x = x
                    self.update_car_pos(a)
                    new_x = self.x
                    t_hat[a][int(x*self.n_xb)][int(new_x*self.n_xb)] += 1

        self.t_hat = np.zeros(shape = (self.actions, self.n_xb, self.n_xb))
        # Normalise
        for a in range(self.actions):
            for i in range(self.n_xb):
                m = np.sum(t_hat[a][i])
                if m > 0:
                    self.t_hat[a][i] = t_hat[a][i] / m

        self.reset()

    def update_car_pos(self, action, step_time = None):
        if not step_time:
            step_time = self.step_time
        # Scale steer from [-1,1] to [-max_steer,max_steer]
        self.steer = self.action_to_steer(action)
        # Add action related noise
        self.steer += abs(self.steer)*np.random.logistic(0, self.action_noise)
        # Add non-action related noise.
        self.steer += np.random.logistic(0, self.steer_noise)
        # Limit steer
        self.steer = min(max(self.steer, -self.max_steer), self.max_steer)
        # Move car
        self.x += self.speed * step_time * math.sin(self.steer)
        # Limit position
        self.x = min(max(self.x, 0),0.999)
        return self.x

    def step(self, action):
        done = False

        if self.training and random.random() < self.attention_switch_prob:
            if self.has_attention == True:
                self.has_attention = False
            else:
                self.has_attention = True

        self.update_car_pos(action)

        self.update_belief(action)

        if self.x < self.threshold or self.x > 1 - self.threshold:
            self.reward = self.oob_reward
        else:
            self.reward = 0

        self.time += self.step_time
        if self.time >= self.max_time:
            done = True

        return self.get_belief(), self.reward, done, {}

# d = driver(17, obs_prob = 0.6)
# d.learn_t_hat()

