import driver
import driver_agent
import search
import search_agent

import numpy as np
from stable_baselines3 import PPO

import matplotlib.pyplot as plt

class multitasking_agent():
    def __init__(self, multitasking):
        self.multitasking = multitasking
        self.agent = PPO("MlpPolicy", self.multitasking, device="cpu", verbose = 0)

    def train_agent(self, max_iters = None, debug = False):
        done = False
        self.multitasking.reset()
        i = 0
        if debug:
            print("Training multitasking agent")
        while not done:
            i += 1
            self.agent.learn(total_timesteps = 1000)
            r, s = self.simulate(120)
            if debug:
                print(round(r,2), round(s,2))

            if max_iters and max_iters == i:
                done = True

    def get_action(self, state = None, deterministic = False):
        if not state:
            state = self.multitasking.get_belief()
        a, _ = self.agent.predict(state, deterministic = deterministic)
        return a

    def simulate(self, until, trace = False, deterministic = False):
        if trace:
            trace = {}
            trace["multi"] = []
            trace["search"] = []
            trace["driver"] = []

        reward = 0
        switches = 0
        self.multitasking.reset()
        while self.multitasking.time < until:
            a = self.get_action(deterministic = deterministic)
            if a != self.multitasking.attended:
                switches += 1
            _, r, _, _ = self.multitasking.step(a)
            if trace:
                trace["multi"].append([self.multitasking.time, a])
                trace["search"].append([self.multitasking.time,
                                        np.sum(self.multitasking.search_agent.search.obs)])
                trace["driver"].append([self.multitasking.time,
                                        self.multitasking.driver_agent.driver.x,
                                        np.sum(self.multitasking.driver_agent.driver.has_attention)])
            reward += r

        if not trace:
            return reward/until, switches/until
        else:
            return trace

def plot_trace(trace, until = 1e10):
    plt.close()
    t = []
    d = []
    d_a = []
    s = []

    for i in range(len(trace["multi"])):
        t.append(trace["multi"][i][0])
        d.append(trace["driver"][i][1])
        d_a.append(trace["driver"][i][2])
        s.append(trace["search"][i][1])
        if trace["multi"][i][0] > until:
            break
    plt.subplot(1, 2, 1)
    plt.scatter(t,d,c = d_a)
    plt.plot(t,d, marker = ".")
    plt.subplot(1, 2, 2)
    plt.scatter(t,s)
    plt.plot(t,s, marker = ".")
    plt.show()

def summarise_trace(trace):
    switch = 0
    attending_driving = True # assume start with driving
    for i in range(len(trace["driver"])):
        if attending_driving and trace["driver"][i][2] == 0:
            switch += 1
            attending_driving = False
        if not attending_driving and trace["driver"][i][2] == 1:
            
            attending_driving = True

    inside = True # note, assuming we start from inside lane
    threshold = 0.2
    oob = 0
    on_road = 1
    off_road = 0
    xs = []
    for i in range(len(trace["driver"])):
        x = trace["driver"][i][1]
        xs.append(x)
        # Only record oob once per violation, not for each step
        if (x < threshold or x > 1 - threshold):
            off_road += 1
        else:
            on_road += 1
        if inside and (x < threshold or x > 1 - threshold):
            oob += 1
            inside = False
        if not inside and (x >= threshold or x <= 1 - threshold):
            inside = True
    std = np.std(xs)

    task_time = []
    start_time = 0
    new_task = True
    for i in range(len(trace["search"])-1):
        if new_task and trace["search"][i+1][1] == 0:
            task_time.append(trace["search"][i+1][0]-start_time)
            start_time = trace["search"][i+1][0]
            new_task = False
        if not new_task and trace["search"][i+1][1] != 0:
            new_task = True

    if len(task_time) == 0:
        task_time = [0]

    ret = {}
    ret['sd_of_x'] = std
    ret['switches'] = switch
    ret['n_oob'] = oob
    ret['oob'] = off_road/(off_road+on_road)
    ret['task_time'] = np.mean(task_time)

    return ret

# d = driver.driver(17)
# d.learn_t_hat()
# d_a = driver_agent.driver_agent(d)
# d_a.train_agent(max_iters = 10)
# s = search.search(3,3)
# s_a = search_agent.search_agent(s)
# s_a.train_agent(max_iters = 10)

# m = multitasking.multitasking(d_a, s_a)
# m_a = multitasking_agent(m)
# m_a.agent = agent
# #m_a.train_agent(max_iters = 10, debug = True)
# trace = m_a.simulate(until = 120, trace = True)
# plot_trace(trace)
