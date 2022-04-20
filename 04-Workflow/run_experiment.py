import driver
import driver_agent
import search
import search_agent
import multitasking
import multitasking_agent

import sys

def run_experiment(params, until, max_iters = 10):
    d = driver.driver(speed = params["speed"], obs_prob = params["obs_prob"],
                           action_noise = params["an"], steer_noise = params["sn"],
                           oob_reward = params["or"])
    d.learn_t_hat()
    d_agent = driver_agent.driver_agent(d)
    d_agent.train_agent(max_iters = max_iters)
    s = search.search(cols = int(params["cols"]),rows = int(params["rows"]), found_reward = params["fr"])
    s_agent = search_agent.search_agent(s)
    s_agent.train_agent(max_iters = max_iters)
    m = multitasking.multitasking(d_agent, s_agent)
    m_agent = multitasking_agent.multitasking_agent(m)
    m_agent.train_agent(max_iters = max_iters)
    trace = m_agent.simulate(until = until, trace = True)
    return trace


# Default params
params = {"speed": 17,
          "obs_prob": 0.8,
          "an": 0.01,
          "sn": 0.02,
          "or": -1,
          "cols": 3,
          "rows": 3,
          "fr": 10}

# Params from command line
# example speed:17 obs_prob:0.8 an:0.01 sn:0.02 or:-1 cols:3 rows:3 fr:10
# heading: speed obs.prob action.noise steering.noise oob.reward cols rows found.reward
# if len(sys.argv) > 2:
#     params_s = None
#     for i in range(len(sys.argv)-1):
#         s = sys.argv[i+1].split(':')
#         assert s[0] in params
#         params[s[0]] = float(s[1])
#         if params_s == None:
#             params_s = s[1]
#         else:
#             params_s = params_s + " " + s[1]
#     until = 1200
#     trace = multitasking_agent.summarise_trace(run_experiment(params, until, max_iters = 20))
#     print(params_s, trace[0]/until, trace[1], trace[2]/until, trace[3]/until)

# def print_params():
#     for s in [17,33]:
#         for obs_prob in [0.8]:
#             for an in [0.01,0.1]:
#                 for sn in [0.01,0.1]:
#                     for or_ in [-5,-1]:
#                         for cols in [3,4]:
#                             for rows in [4,4]:
#                                 for fr in [1, 10]:
#                                     st = "speed:" + str(s) + " obs_prob:" + str(obs_prob) + " an:" + str(an) + " sn:" + str(sn) + " or:" + str(or_) + " cols:" + str(cols) + " rows:" + str(rows) + " fr:" + str(fr)
#                                     print(st)

# if len(sys.argv) == 2:
#     print_params()
