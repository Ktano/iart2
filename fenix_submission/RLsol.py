# Grupo 081; Pedro Caetano 56564; Francisco Henriques 75278;

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:31:54 2017

@author: mlopes
"""


import numpy as np
import matplotlib.pyplot as plt

def Q2pol(Q, eta=5):
    e = np.exp(eta*Q)
    ones = np.ones((2, 2))
    return e / e.dot(ones)
	
class myRL:

    def __init__(self, nS, nA, gamma):
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.Q = np.zeros((nS,nA))
        
    def traces2Q(self, trace, alpha=.01, err_threshold=1e-4):

        def unpack_transition(transition):
            s0, a, s1 = map(int, transition[:3])
            r = transition[3]
            return s0, a, r, s1 #current_state, action, reward, hypothetical_next_state

        gamma = self.gamma #discount factor
        Q = self.Q.copy()
        
        while True:            
            for transition in trace:
                s0, a, r, s1 = unpack_transition(transition)

                l = r + gamma * Q[s1].max() #learned value

                Q[s0,a] = (1-alpha) * Q[s0,a] + alpha * l #update q-value

            err = np.linalg.norm(self.Q - Q)

            self.Q = Q.copy()

            if err<err_threshold:
                break 

        return self.Q


def plot(axes, filename, xlabel = "Steps", ylabel = "States"):

    axes.set_xlim(x.min()-1, x.max()+1)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(axis='y')

    plt.tight_layout()

    fig.savefig(filename)


def plot_trajectory(traj, name):

    y = np.concatenate([traj[:, 0],  traj[-1, -1:]])

    x = np.arange(len(y))

    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(111)

    ax.plot(x, y, 'ro:', linewidth=.5, markersize=1.)

    ax.legend([f"Trajectory for {name}"])

    plot(ax, f"{name}.png")

def plot_trajectories(*trajs, colors='rgb', fn='trajectories', ff='png'):

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    for traj,c in zip(trajs, colors):

        y = np.concatenate([traj[:, 0],  traj[-1, -1:]])
        x = np.arange(len(y))
        ax.plot(x, y, f'{c}o:', linewidth=.5, markersize=1.)

    ax.legend([f"Trajectory {i}" for i in range(1, len(trajs)+1)])

    plot(ax, f"{fn}.{ff}")
