# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:31:54 2017

@author: mlopes
"""

# Grupo 081; Pedro Caetano 56564; Francisco Henriques 75278;

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
            indices = [0, 1, 3, 2]
            return transition[indices]

        gamma = self.gamma
        Q = self.Q.copy()
        
        while True:            
            for transition in trace:
                s0, a, r, s1 = unpack_transition(transition)
                s0, a, s1 = map(int, (s0, a, s1))

                q = Q[s0,a] # q utility/value

#                alpha = 1 / (1+q)#                gamma = pow(self.gamma, i);i+=1
#                tde = r + gamma * Q[s1].max() - q #temporal difference error
#                Q[s0,a] += alpha * tde

                l = r + gamma * Q[s1].max() #learned value

                Q[s0,a] = (1-alpha) * q + alpha * l

            err = np.linalg.norm(self.Q - Q)

            self.Q = Q.copy()

            if err<err_threshold:
                break 

        return self.Q


def plot_trajectory(traj, name, xlabel = "Steps", ylabel = "States"):

    y = np.concatenate([traj[:, 0],  traj[-1, -1:]])

    x = np.arange(len(y))

    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(111)

    ax.plot(x, y, 'ro:', linewidth=.5, markersize=1.)

    ax.legend([f"Trajectory for {name}"])

#    plt.xticks(x)
    left=0.038
    right=0.983
    d = dict(left=left,right=right)
#    plt.subplots_adjust(**d)
    ax.set_xlim(x.min()-1, x.max()+1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y')

    plt.tight_layout()
    fig.savefig(f"{name}.png")

def plot_trajectories(*trajs, colors='rgb', xlabel = "Steps", ylabel = "States", fn='trajectories', ff='png'):

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    for traj,c in zip(trajs, colors):

        y = np.concatenate([traj[:, 0],  traj[-1, -1:]])
        x = np.arange(len(y))
        ax.plot(x, y, f'{c}o:', linewidth=.5, markersize=1.)

    ax.legend([f"Trajectory {i}" for i in range(1, len(trajs)+1)])

#    plt.xticks(x)
    left=0.038
    right=0.983
    d = dict(left=left,right=right)
#    plt.subplots_adjust(**d)
    ax.set_xlim(x.min()-1, x.max()+1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y')

    plt.tight_layout()
    
    fig.savefig(f"{fn}.{ff}")
