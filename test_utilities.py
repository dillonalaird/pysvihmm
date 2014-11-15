from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def mistake_rate(sts, var_q):
    incor = 0
    for i in xrange(len(sts)):
        pred = np.argmax(var_q[i,:])
        if pred != sts[i]:
            incor += 1

    return incor / float(len(sts))

def plot_MAP(var_q, obs):
    x = [i for i in range(len(obs))]
    plt.plot(x, obs)

    colors = ['r', 'b', 'g']
    p = np.argmax(var_q[0])
    s = 0
    for i in xrange(1, len(obs)):
        n = np.argmax(var_q[i])
        if n != p:
            plt.axvspan(s, i - 1, facecolor=colors[p], alpha=0.5)
            s = i - 1
            p = n

    plt.axvspan(s, len(obs) - 1, facecolor=colors[p], alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Feature Value')
    plt.show()

def plot_bar(var_q, obs):
    colors = ['r', 'b', 'g']

    colormap = []
    bounds = [0]
    p = np.argmax(var_q[0])
    for i in xrange(1, len(obs)):
        n = np.argmax(var_q[i])
        if n != p:
            colormap.append(colors[p])
            bounds.append(i)
            p = n

    #cmap = mpl.colors.ListedColormap(colormap)
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

def plot_var_q(var_q, sts):
    t = np.arange(len(sts))
    var_q_true = []
    for i in xrange(len(sts)):
        var_q_true.append(var_q[i][sts[i]])

    plt.fill_between(t, var_q_true)
    plt.xlabel('Time')
    plt.ylabel('Probability of True State')
    plt.show()


