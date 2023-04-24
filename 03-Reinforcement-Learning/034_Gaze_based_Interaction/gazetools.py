
import os
import csv

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from IPython import display

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


output_dir = 'output/'
policy_file = 'policy'


def train(env, timesteps):
    ''' 
    '''
    env = Monitor(env, output_dir)
    controller = PPO('MlpPolicy', env, verbose=0, clip_range=0.15)
    controller.learn(total_timesteps=int(timesteps))
    controller.save(f'{output_dir}{policy_file}')
    print('Done training.')
    return controller

'''
    Plot learning curve
''' 

def plot_learning_curve(title='Learning Curve'):
    """
    :param output_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(output_dir), 'timesteps')
    y = moving_average(y, window=100)
    # Truncate x
    #x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

'''
    Get the Euclidean distance between points p and q 
'''

def get_distance(p,q):
    return np.sqrt(np.sum((p-q)**2))

'''
'''

def load_controller():
    controller = PPO.load(f'{output_dir}{policy_file}')
    return controller

'''
'''

def run_model(env, controller, n_episodes, filename):
    '''
    run the model for n_episodes and save its behaviour in a csv file.
    Note that 'env' is a term used by Gym to describe everything but the controller.
    '''
    max_episodes = 900000
    if n_episodes > max_episodes:
        print(f'We ask that you limit training to a max of {max_episodes} on the School of Computer Science AWS account.')
        print(f'If you want to run more training episodes then please do so on a local computer.')
        return

    result = []
    # repeat for n episodes
    eps = 0
    while eps < n_episodes:                
        done = False
        step = 0
        obs, _ = env.reset()
        # record the initial state
        info = env.get_info()
        info['episode'] = eps
        result.append(info)
 
        # repeat until the gaze is on the target.
        while not done:
            step+=1
            # get the next prediction action from the controller
            action, _ = controller.predict(obs,deterministic = True)
            obs, reward, done, _, info = env.step(action)
            info['episode'] = eps
            result.append(info)
            if done:
                eps+=1
    path = f'{output_dir}{filename}'
    df = pd.DataFrame(result)
    df.to_csv(path,index=False)
    return df

'''
'''

def plot_gaze(gaze_x,gaze_y):
    plt.plot(gaze_x,gaze_y,'r+',markersize=20,linewidth=2)

'''
'''

def update_display(gap_time):
    # update the display with a time step
    display.display(plt.gcf())
    display.clear_output(wait=True)
    time.sleep(gap_time)

'''
'''

def set_canvas():
    time.sleep(2)  
    #set the canvas
    plt.close()
    fig, ax = plt.subplots(figsize=(7,7)) # note we must use plt.subplots, not plt.subplot
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.gca().set_aspect('equal', adjustable='box')
    update_display(gap_time=1)
    return(ax)

'''
'''

def animate_episode(ax, df_eps):
    gaze_x,gaze_y=df_eps.loc[0]['fixate_x'],df_eps.loc[0]['fixate_y']
    for t in range(1,len(df_eps)):
        # each time step t

        if t==2:
            # target
            target_x,target_y=df_eps.loc[t]['target_x'],df_eps.loc[t]['target_y']
            target_width=df_eps.loc[t]['target_width']
            circle1 = plt.Circle((target_x,target_y), target_width/2, color='k')
            ax.add_patch(circle1)
            update_display(gap_time=0.5)

        new_gaze_x,new_gaze_y=df_eps.loc[t]['fixate_x'],df_eps.loc[t]['fixate_y'] 
        plt.arrow(gaze_x,gaze_y, new_gaze_x-gaze_x,new_gaze_y-gaze_y,head_width=0.05,
                      length_includes_head=True,linestyle='-',color='r')
        plot_gaze(new_gaze_x,new_gaze_y)
        update_display(gap_time=0.5)
            
         # new gaze becomes the current gaze
        gaze_x,gaze_y=new_gaze_x,new_gaze_y

'''
'''

def animate_multiple_episodes(data, n):
    for eps in range(n):
        # each episode
        ax = set_canvas()
    
        # behaviour data for each episode
        df_eps=data.loc[data['episode']==eps]
        df_eps.reset_index(drop=True, inplace=True)
    
        # truncate the length of the episode if it is too long.
        if len(df_eps) > 5:
            df_eps = df_eps[0:5]
        
        animate_episode(ax, df_eps)

