from gym import Env
from gym.spaces import Discrete, Dict, Box

from stable_baselines3 import PPO

import numpy as np

#import carla

class driver_agent(Env):
    # carla argument should be an initialised carla client
    def __init__(self, carla_env, goal_reward = 1, collision_reward = -1):
        self.carla_env = carla_env

        self.goal_reward = goal_reward
        self.collision_reward = collision_reward

        # go / no go
        self.action_space = Discrete(2)

        self.observation_space = Dict(
            spaces = {
                "distance": Box(0, 1, (1,)),
                "speed": Box(0, 1, (1,)),
                "acceleration": Box(0, 1, (1,))
            })

        self.agent = PPO("MultiInputPolicy", self, verbose = 1)

    def reset(self):
        self.carla_env.reset()
        #print("reset")
                
        self.done = False
        self.ticks = 0

        self.belief = self.carla_env.get_state()

        return self.belief

    def step(self, action):
        #print(action)
        self.reward = 0
        # action: no go
        if action == 0:
            self.carla_env.tick()
            self.ticks += 1
            # Break if nothing ever happens
            if self.ticks > 1000:
                print("Too many ticks")
                self.done = True
        # action: go
        if action == 1:
            self.done = True
            collision, _ = self.carla_env.simulate_go()
            if collision:
                self.reward = -1
            else:
                self.reward = 1

        self.belief = self.carla_env.get_state()
        #print(self.belief)
        #print(self.reward)

        return self.belief, self.reward, self.done, {}

    def run_episode(self, render = False):
        if render and self.carla_env.settings.no_rendering_mode:
            self.carla_env.settings.no_rendering_mode = False
            self.carla_env.world.apply_settings(self.carla_env.settings)
        self.reset()
        ticks = 0

        while not self.done:
            ticks += 1
            a = self.agent.predict(self.belief)[0]
            self.step(a)

        # if render:
        #     self.carla_env.settings.no_rendering_mode = False
        #     self.carla_env.world.apply_settings(self.carla_env.settings)
            
        return ticks, self.reward

    def train_agent(self, total_timesteps = 1000, iters = 10):
        self.agent.learn(total_timesteps = total_timesteps)
        for i in range(iters):
            self.agent.learn(total_timesteps = total_timesteps)
            ts = []
            rs = []
            for j in range(10):
              t, r = self.run_episode()
              ts.append(t)
              rs.append(r)
            print(np.mean(ts), np.mean(rs))
                
