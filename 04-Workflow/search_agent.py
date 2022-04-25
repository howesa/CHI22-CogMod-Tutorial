import search

import numpy as np
from stable_baselines3 import PPO

class search_agent():
    def __init__(self, search):
        self.search = search
        self.agent = PPO("MlpPolicy", self.search, verbose = 0)

        # Make a record of min and max values for scaling predicted
        # value between [0,1]
        self.min_value = None
        self.max_value = None

    def predict_value(self, scale = False):
        values = self.agent.policy.forward(self.agent.policy.obs_to_tensor(self.search.obs)[0])
        if scale:
            return (values[1].item()-self.min_value)/(self.max_value-self.min_value)
        else:        
            return values[1].item()

    def run_episode(self, task_type = None):
        self.search.reset()
        if task_type != None:
            assert task_type >= 0 and task_type < 3
            while self.task_type != task_type:
                self.search.randomise_search_device()
        tr = []
        while not self.search.done:
            val = self.predict_value()
            if self.max_value == None or val > self.max_value:
                self.max_value = val
            if self.min_value == None or val < self.min_value:
                self.min_value = val                            
            action = self.get_action()
            self.search.step(action)
            tr.append([self.search.mt, self.search.action_to_loc(action)])
        return self.search.mt, self.search.steps, self.search.fixations, tr

    def get_action(self):
        return self.agent.predict(self.search.obs)[0]

    def run_experiment(self, episodes = 100):
        mt = []
        steps = []
        fixations = []
        for i in range(episodes):
            m, s, f, _ = self.run_episode()
            mt.append(m)
            steps.append(s)
            fixations.append(f)
        return np.mean(mt), np.mean(steps), np.mean(fixations)

    def train_agent(self, debug = False, max_iters = 10):
        training_info = []
        timesteps = 0
        timesteps_increment = 10000
        done = False
        i = 0
        while not done:
            i += 1
            self.agent.learn(total_timesteps = timesteps_increment)
            timesteps += timesteps_increment
            m, s, f = self.run_experiment()
            training_info.append([timesteps, m, s, f])
            if debug:
                print(round(m,2), round(s,1), round(f,1))
            if i == max_iters:
                done = True

        return training_info

def visualise_training(data, col):
    plt.close()
    x = []
    y = []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][col])
    plt.scatter(x, y)
    plt.show()

# s = search.search(3,3)
# a_s = search_agent(s)
# info = a_s.train_agent()
