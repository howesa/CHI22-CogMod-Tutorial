import importlib

import matplotlib.pyplot as plt
import numpy as np

import driver_agent_physics
import physics_env
import animate_trace

importlib.reload(driver_agent_physics)
importlib.reload(physics_env)
importlib.reload(animate_trace)

def make_agent(sigma = 0, iters = 10):
    e = physics_env.physics_env()
    agent = driver_agent_physics.driver_agent_physics(e, observation_var = sigma)
    agent.max_distance = 30
    agent.penalty_per_tick = 0.1
    e.veh1_straight_start_y_range = [-25,-2]
    agent.train_agent(total_timesteps = 10000, iters = iters)
    return agent

def retrain_agent(agent, y_range = None, iters = 5):
    if y_range:
        agent.env.veh1_straight_start_y_range = [y_range[0],y_range[1]]
    agent.train_agent(total_timesteps = 10000, iters = iters)
    agent.env.veh1_straight_start_y_range = [-25,-2]

def animate_agent(agent, y_start = None, get_anim = True, x_lim = [-50,50], y_lim = [50,-50]):
    agent.env.save_trace = True
    print(agent.run_episode(y_start = y_start))
    agent.env.save_trace = False
    return animate_trace.animate_trace(agent.env.trace, get_anim = get_anim, x_lim = x_lim, y_lim = y_lim)

def wait_or_go_experiment(agent, y_range, n = 100):
    data = []
    a.env.veh1_straight_start_y_range = y_range
    for i in range(n):
        _, _, _, w, c = a.run_episode()
        data.append([a.observation_var, a.env.y_start, w, c])
    
    agent.env.veh1_straight_start_y_range = [-25,-2]
    return data



# Visualize the probability of go/no go as the function of y_start, between different sigmas.
# Note that the lines may dip close to max y_start, this is an artefact of the smooting.

# from scipy.stats import gaussian_kde

# def estimate_probability(df, sigma, y_start_values):
#     sub_df = df[df['sigma'] == sigma]
#     kde = gaussian_kde(sub_df[['y_start', 'wait']].T)
#     probabilities = kde.evaluate(np.column_stack((y_start_values, np.ones_like(y_start_values))).T)
#     return probabilities

# def plot_probability_lines(df):
#     fig, ax = plt.subplots()

#     y_start_range = np.linspace(df['y_start'].min(), df['y_start'].max(), num=500)

#     for sigma in df['sigma'].unique():
#         probabilities = estimate_probability(df, sigma, y_start_range)
#         ax.plot(y_start_range, probabilities, label=f'Sigma: {sigma}')

#     ax.set_xlabel('y_start')
#     ax.set_ylabel('Probability of Wait')
#     ax.legend(title="Sigma", loc="upper left")
#     plt.title('Probability of wait across y_start by sigma')
#     plt.show()
