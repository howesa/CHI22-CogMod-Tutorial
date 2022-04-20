import driver

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import SAC

import matplotlib.pyplot as plt

class driver_agent():
    def __init__(self, driver):
        self.driver = driver
        if self.driver.continuous:
            self.agent = SAC("MlpPolicy", self.driver, device="cpu", verbose = 0)
        else:
            self.agent = PPO("MlpPolicy", self.driver, device="cpu", verbose = 0)
        self.driver.agent = self.agent

        # Make a record of min and max values for scaling predicted
        # value between [0,1]
        self.min_value = None
        self.max_value = None

    def train_agent(self, max_iters = None, debug = False):

        done = False
        convergence = 0
        rs = []
        previous_r = None
        self.driver.training = True
        self.driver.reset()
        i = 0
        if debug:
            print("Training driver agent")
        while not done:
            i += 1
            self.agent.learn(total_timesteps = 1000)
            r = self.simulate(120)
            rs.append(r)
            if debug:
                if len(rs) > 5:
                    print(round(r, 2), np.std(rs), np.std(rs[len(rs)-5:])/np.std(rs))
                else:
                    print(round(r, 2), np.std(rs))

            previous_r = r
            if convergence == 4:
                done = True
            if max_iters and max_iters == i:
                done = True
        self.driver.training = False

    def predict_value(self, scale = False):
        values = self.agent.policy.forward(self.agent.policy.obs_to_tensor(self.driver.get_belief())[0])
        if self.driver.continuous:
            values = values.item()
        else:
            values = values[1].item()
        if scale:
            return (values-self.min_value)/(self.max_value-self.min_value)
        else:        
            return values

    def plot_value(self, H):
        x = np.linspace(0,0.999,self.driver.n_xb)
        y = []
        for x_i in x:
            y.append(self.predict_value(state = [x_i, H]))
        plt.close()
        plt.scatter(x,y)
        plt.show()

    def plot_value_H(self, x):
        H = np.linspace(0,1,100)
        y = []
        for h_i in H:
            y.append(self.predict_value(state = [x, h_i]))
        plt.close()
        plt.scatter(H,y)
        plt.show()        

    def get_action(self, state = None):
        if not state:
            state = self.driver.get_belief()
        a, _ = self.agent.predict(state)
        return a

    def simulate(self, until):
        rewards = 0
        self.driver.reset()
        while self.driver.time < until:
            val = self.predict_value()
            if self.max_value == None or val > self.max_value:
                self.max_value = val
            if self.min_value == None or val < self.min_value:
                self.min_value = val                
            action = self.get_action()
            _, r, _, _ = self.driver.step(action)
            rewards += r
        return rewards/until

# d = driver.driver(17, obs_prob = 0.8, continuous = True)
# d.learn_t_hat()
# #d.t_hat = t_hat
# a = driver_agent(d)
# a.train_agent(debug = True)
