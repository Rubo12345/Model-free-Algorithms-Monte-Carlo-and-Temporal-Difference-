#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
import itertools
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    policy = np.argmax(Q[state])
    greedy_prob = np.ones(nA)*epsilon/nA
    greedy_prob[policy] += 1 - epsilon
    action = np.random.choice(np.arange(len(Q[state])),p = greedy_prob)
    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    nA = env.action_space.n
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop n_episodes
    for episode in range(0,n_episodes):
        # define decaying epsilon
        if episode > 0:
            epsilon = 0.99*epsilon
        # initialize the environment 
        current_obs = env.reset()
        # get an action from policy
        action = epsilon_greedy(Q, current_obs, nA, epsilon)
        # loop for each step of episode
        for t in itertools.count():
            # return a new state, reward and done
            new_obs,r,done,info,prob = env.step(action)
            # get next action
            next_action = epsilon_greedy(Q, new_obs, nA, epsilon)
            # TD update
            # td_target
            td_target = r + gamma * Q[new_obs][next_action]
            # td_error
            td_error = td_target - Q[current_obs][action]
            # new Q
            Q[current_obs][action] += alpha * td_error
            if done:
                break
            # update state
            current_obs = new_obs
            # update action
            action = next_action
    # print(Q)
    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    nA = env.action_space.n
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop n_episodes
    for episode in range(0,n_episodes):
        # initialize the environment 
        current_obs = env.reset()
        # loop for each step of episode
        for t in itertools.count():
            # get an action from policy
            action = epsilon_greedy(Q, current_obs, nA, epsilon)
            # return a new state, reward and done
            new_obs,r,done,info,prob = env.step(action)
            # TD update
            # td_target with best Q
            td_target = r + gamma * Q[new_obs][np.argmax(Q[new_obs])]
            # td_error
            td_error = td_target - Q[current_obs][action]
            # new Q
            Q[current_obs][action] += alpha * td_error
            if done:
                break
            # update state
            current_obs = new_obs
    ############################
    return Q