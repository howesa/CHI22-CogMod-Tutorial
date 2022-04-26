# Search model

from gym import Env
from gym.spaces import Discrete, Dict, Box
import numpy as np
import random
import math

class search(Env):
    def __init__(self, rows = 3, cols = 3, found_reward = 10):
        self.rows = rows
        self.cols = cols
        self.cell_distance = 5   # visual degrees
        self.found_reward = found_reward
        self.respond_time = 0.15 # time to respond to a task

        # The visual search model calculates a lot of distances with a fixed
        # number of parameters. Makes it faster to tabulate.
        self.distances = {}

        self.action_space = Discrete(rows*cols)
        self.observation_space = Box(0, 1, (rows,cols), dtype = int)

        self.log = []

        self.task = 0

        self.randomise_task_type = False

        self.agent = None

        self.reset()

    def reset(self):
        self.done = False
        self.task += 1
        self.steps = 0
        self.fixations = 0

        self.randomise_search_device()
        self.obs = obs = np.zeros(shape = self.observation_space.shape, dtype = int)
        self.found = False
        self.reward = 0
        self.mt = 0
        self.moved = True

        # Start with a random eye location but do not observe that location. Reward for the start is 0.
        self.eye_loc = self.action_to_loc(self.action_space.sample())

        self.reward = 0

        return self.obs

    def get_belief(self):
        return self.obs

    def randomise_search_device(self):
        # Task type. 0 means that the search target is randomly among
        # the items. 1 means that the target is always the last
        # item that is looked at.
        if self.randomise_task_type:
            self.task_type = random.choice([0,1])
        else:
            self.task_type = 0

        if self.task_type == 1:
            self.target = self.action_to_loc(self.action_space.sample())
        else:
            self.target = None

    # Calculate visual distance, as degrees, between two screen
    # coordinates. User distance needs to be given in the same unit.
    def visual_distance(self, start, end, user_distance):
        distance = math.sqrt(pow(start[0] - end[0], 2) + pow(start[1] - end[1],2))
        return 180 * (math.atan(distance / user_distance) / math.pi)

    # Eye movement and encoding time come from EMMA (Salvucci, 2001). Also
    # return if a fixation occurred.
    def EMMA_fixation_time(self, distance, freq = 0.1, encoding_noise = False):
        emma_KK = 0.006
        emma_k = 0.4
        emma_prep = 0.135
        emma_exec = 0.07
        emma_saccade = 0.002
        E = emma_KK * -math.log(freq) * math.exp(emma_k * distance)
        if encoding_noise:
            E += np.random.gamma(E, E/3)
        if E < emma_prep: return E, False
        S = emma_prep + emma_exec + emma_saccade * distance
        if (E <= S): return S, True
        E_new = (emma_k * -math.log(freq))
        if encoding_noise:
            E_new += np.random.gamma(E_new, E_new/3)
        T = (1 - (S / E)) * E_new
        return S + T, True

    # Euclidian distance between two points. Use lookuptable for reference
    # or update it with a new entry.
    def distance(self, x1,y1,x2,y2):
        if (x1,y1,x2,y2) not in self.distances:
            self.distances[(x1,y1,x2,y2)] = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2,2))
        return self.distances[(x1,y1,x2,y2)]

    def step(self, action):
        self.steps += 1
        mt = 0
        self.reward = 0
        self.done = False

        loc = self.action_to_loc(action)

        # Move eyes, calculate mt.
        eccentricity = self.distance(self.eye_loc[0], self.eye_loc[1], loc[0], loc[1])
        mt, moved = self.EMMA_fixation_time(eccentricity*self.cell_distance, encoding_noise = False)
        if moved:
            self.eye_loc = loc
            self.fixations += 1
        self.obs[loc[0]][loc[1]] = 1
        # Calculate reward and mark if target found.
        if self.task_type == 1:
            if loc == self.target:
                mt += self.respond_time
                self.reward = self.found_reward
                self.done = True
        elif self.obs.all() == True:
            mt += self.respond_time
            self.reward += self.found_reward
            self.done = True

        self.reward -= mt
        self.mt += mt
        self.moved = moved

        return self.obs, self.reward, self.done, {}

    # Given a discrete action, what is the corresponding visual location in the matrix?
    def action_to_loc(self, action):
        return [int(round(np.floor(action / self.cols))), round(action % self.cols)]


# s = search(4,3)
# info = s.train_agent()
