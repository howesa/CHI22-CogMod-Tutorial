from gym import Env
from gym.spaces import Discrete, Dict, Box
from copy import copy

from stable_baselines3 import PPO

import numpy as np
import math

class driver_agent_physics(Env):
    def __init__(self, physics_env, goal_reward = 1, collision_reward = -1, observation_var = 0):
        self.env = physics_env

        self.goal_reward = goal_reward
        self.collision_reward = collision_reward

        self.max_ticks = 256
        self.max_distance = 80

        # go / no go
        self.action_space = Discrete(2)        

        self.observation_space = Dict(
            spaces = {
                "distance": Box(0, 1, (1,)),
                "passed": Box(0, 1, (1,)),
                "distance_var": Box(0, 1, (1,)),
                "speed": Box(0, 1, (1,)),
                "acceleration": Box(0, 1, (1,)),
                "ticks": Box(0, 1, (1,))
            })

        self.penalty_per_tick = 0

        self.observation_var = observation_var

        self.agent = PPO("MultiInputPolicy", self, device = 'cuda',
                         learning_rate=0.00025,
                         ent_coef=0.01,
                         # n_steps = 128,
                         # batch_size = 64,
                         verbose = 0)

    def reset(self, y_start = None):
        self.env.reset(y_start = y_start)

        self.done = False
        self.ticks = 0

        self.collision = False
        self.distance_at_go = None
        self.waited_before_go = False

        # Init prior as basically flat.
        self.prior = {}
        self.prior['distance'] = self.max_distance/2
        self.prior['distance_var'] = 1000

        self.belief = self.get_belief()

        return self.belief

    def step(self, action):
        self.reward = 0
        # action: no go
        if action == 0:
            self.env.tick()
            self.ticks += 1
            # break if nothing ever happens
            if self.ticks > self.max_ticks:
                #print("Too many ticks")
                self.reward = -10
                self.done = True
            if self.env.get_distance() > self.max_distance:
                self.reward = -10
                self.done = True                
        # action: go
        if action == 1:
            # Did we wait for the other car before going?
            if self.env.veh2_turn_pos[1] < self.env.veh1_straight_pos[1]:
                self.waited_before_go = True
            self.distance_at_go = self.env.get_distance()
            self.done = True
            self.collision, _ = self.env.simulate_go()
            if self.collision:
                self.reward = -10
            else:
                self.reward = 10 - self.penalty_per_tick * self.ticks

        self.belief = self.get_belief()

        return self.belief, self.reward, self.done, {}

    def get_belief(self):
        s = self.env.get_state()
        s['passed'] = [0] if self.env.veh2_turn_pos[1] > self.env.veh1_straight_pos[1] else [1]
        # Make observation noisy, as in https://github.com/gmarkkula/COMMOTIONSFramework
        D = s['distance']
        d_oth = np.linalg.norm(self.env.veh1_straight_pos-[-1.825, -1.506512])
        h = 1.5
        s['distance_var'] = d_oth * (1 - h / (D*math.tan(math.atan(h/D) + self.observation_var)))
        s['distance_var'] = max(0, s['distance_var'])
        s['distance'] = np.random.normal(s['distance'], s['distance_var'])
        if self.observation_var > 0:
            s['distance'], s['distance_var'] = \
                self.kalman_update(self.prior['distance'],
                                   self.prior['distance_var'],
                                   np.random.normal(s['distance'],
                                                    self.observation_var),
                                   self.observation_var)
            self.prior['distance'] = s['distance']
            self.prior['distance_var'] = s['distance_var']
        # Normalise into arrays
        # TODO: Set maxes as constants and obey them all over
        s['ticks'] = [self.ticks/self.max_ticks]
        if self.observation_var > 0:
            s['distance_var'] = [s['distance_var'] / (self.observation_var**2)]
        else:
            s['distance_var'] = [0]
        s['speed'] = [s['speed'] / 30]
        s['distance'] = [s['distance'] / (4*self.max_distance)]
        s['acceleration'] = [s['acceleration']]

        if s['speed'][0] > 1 or s['distance'][0] > 1 or s['acceleration'][0] > 1 or s['distance_var'][0] > 1:
            print("Box overflow")
            print(self.env.get_state())
            print(s)

        return s

    def run_episode(self, render = False, deterministic = False, y_start = None):
        self.agent.policy.set_training_mode(False)
        # if render and self.carla_env.settings.no_rendering_mode:
            # self.carla_env.settings.no_rendering_mode = False
            # self.carla_env.world.apply_settings(self.carla_env.settings)
        self.reset(y_start = y_start)
        ticks = 0
        total_reward = 0

        while not self.done:
            ticks += 1
            a = self.agent.predict(self.belief, deterministic = deterministic)[0]
            self.step(a)
            total_reward += self.reward

        # if render:
        #     self.carla_env.settings.no_rendering_mode = False
        #     self.carla_env.world.apply_settings(self.carla_env.settings)

        self.agent.policy.set_training_mode(True)

        return ticks, total_reward, self.distance_at_go, 1 if self.waited_before_go else 0, 1 if self.collision else 0

    def train_agent(self, total_timesteps = 1000, iters = 10, print_debug = True):
        # i = iter
        # t = ticks
        # r = reward
        # d = distance at go
        # w = waited for the other car before go
        # c = collisions
        if print_debug:
            print("\ni\tt\tr\td\tw\tc")
        for i in range(iters):
            self.agent.learn(total_timesteps = total_timesteps)
            self.run_episodes(100, prefix = str(i), print_debug = print_debug)

    def run_episodes(self, n, prefix = "0", print_debug = True):
        ts = []
        rs = []
        ds = []
        ws = []
        cs = []
        for i in range(n):
            t, r, d, w, c = self.run_episode()
            ts.append(t)
            rs.append(r)
            if d: ds.append(d)
            ws.append(w)
            cs.append(c)
        if len(ds) == 0: ds = [0] # to avoid warning in cases where zero gos were observed
        if print_debug:
            print(prefix, "\t", round(np.mean(ts),2), "\t", round(np.mean(rs),2), "\t", round(np.mean(ds),2), "\t", round(np.mean(ws),2), "\t", round(np.mean(cs),2), sep='')


    def kalman_update(self, prior_mean, prior_var, observation, s):
        """
        Update the belief about x using the Kalman filter.

        Parameters:
        prior_mean (float): The mean of the prior belief about x.
        prior_var (float): The variance of the prior belief about x.
        observation (float): The noisy observation of x.
        s (float): The standard deviation of the observation noise.

        Returns:
        float, float: The mean and variance of the posterior belief about x.
        """

        # The observation model has a gain of 1 (linear relationship)
        observation_gain = 1

        # Calculate the Kalman gain
        kalman_gain = prior_var * observation_gain / (prior_var * observation_gain**2 + s**2)

        # Update the mean and variance of the belief about x
        posterior_mean = prior_mean + kalman_gain * (observation - prior_mean)
        posterior_var = (1 - kalman_gain * observation_gain) * prior_var

        return posterior_mean, posterior_var
