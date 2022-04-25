
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class multitasking(Env):
    def __init__(self, driver_agent, search_agent):
        self.driver_agent = driver_agent
        self.search_agent = search_agent

        self.action_space = Discrete(2)
        self.observation_space = Box(0,1, (3,), dtype = float)

        self.attention_switch_time = 0.15

        self.reset()

    def reset(self):
        self.driver_agent.driver.reset()
        self.search_agent.search.reset()

        self.done = False

        self.next_driver_step = 0

        self.time = 0

        self.attended = 0

        return self.get_belief()

    def get_belief(self):
        return [self.attended, self.driver_agent.predict_value(scale = True), self.search_agent.predict_value(scale = True)]

    def step(self, action):
        mt = 0
        self.reward = 0

        # Switch attention?
        if action != self.attended:
            if self.attended == 0:
                self.attended = 1
            else:
                self.attended = 0
            # Drive blindly for the duration of the attention switch
            self.driver_agent.driver.has_attention = False
            for i in range(int(np.ceil(self.attention_switch_time / self.driver_agent.driver.step_time))):
                a = self.driver_agent.get_action()
                self.driver_agent.driver.step(a)
            mt += self.attention_switch_time
            self.next_driver_step = self.time + self.attention_switch_time

        # Drive?
        if action == 0:
            self.driver_agent.driver.has_attention = True
            a = self.driver_agent.get_action()
            self.driver_agent.driver.step(a)
            mt += self.driver_agent.driver.step_time
            # Record next step in case visual search goes on longer
            self.next_driver_step += self.driver_agent.driver.step_time

        # Search?
        if action == 1:
            mt_ = self.search_agent.search.mt
            a = self.search_agent.get_action()
            self.search_agent.search.step(a)
            mt += self.search_agent.search.mt-mt_

            # Drive blindly?
            if self.time >= self.next_driver_step:
                a = self.driver_agent.get_action()
                self.driver_agent.driver.step(a)
                self.next_driver_step += self.driver_agent.driver.step_time

        self.time += mt

        self.reward = self.driver_agent.driver.reward + self.search_agent.search.reward

        if self.search_agent.search.done:
            self.search_agent.search.reset()

        return self.get_belief(), self.reward, self.done, {}

