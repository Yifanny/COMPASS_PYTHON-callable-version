import math
import numpy as np
import numpy.matlib as npm
import random
import copy

import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
from mpl_toolkits.mplot3d import Axes3D
import time

import util

#Pick a direction uniformly
def direction(d):
    z = np.random.normal(size = d)
    denominator = 0
    for i in range(d):
        denominator += z[i]**2
        if z[i] < 0:
            z[i] = -z[i]
    denominator = 1/math.sqrt(denominator)
    return denominator * z

#Determin the upperbound and lowerbound
def ub_lb(x_star, Vy, cons, scale=1):
    m, d = Vy.shape
    c = util.read_constraint(cons)
    
    #determin direction
    z = direction(d)
    
    #determin the most promising area
    Vy = util.remove(Vy, util.ismember(x_star, Vy))
    t1 = util.mat_sub(np.matlib.repmat(x_star, m-1, 1), Vy)
    t2 = util.mat_add(np.matlib.repmat(x_star, m-1, 1), Vy) * 0.5
    
    #initialize upperbound and lowerbound
    ub = [0] * d
    lb = [0] * d
    for i in range(d):
        ub[i] = x_star[i] + z[i] * scale
        lb[i] = x_star[i] - z[i] * scale

    #determin upper bound
    check_ub1 = np.dot(t1.T, util.mat_sub(np.matlib.repmat(ub, m-1, 1), t2))
    check_ub2 = True
    while (np.sum(util.mat_abs(check_ub1)) >= 1e-6) and check_ub2 :
        for i in range(d):
            if ub[i] > c[i][1]:
                check_ub2 = False
                break
            else:
                ub[i] += z[i] * scale
        check_ub1 = np.dot(t1.T, util.mat_sub(np.matlib.repmat(ub, m-1, 1), t2))
    
    for j in range(d):
        ub[j] -= z[j] * scale

    #determin lower bound
    check_lb1 = np.dot(t1.T, util.mat_sub(np.matlib.repmat(lb, m-1, 1), t2))
    check_lb2 = True
    while (np.sum(util.mat_abs(check_lb1)) >= 1e-6) and check_lb2 :
        for i in range(d):
            if lb[i] < c[i][0]:
                check_lb2 = False
                break
            else:
                lb[i] -= z[i] * scale
        check_lb1 = np.dot(t1.T, util.mat_sub(np.matlib.repmat(lb, m-1, 1), t2))
        
    for j in range(d):
        lb[j] += z[j] * scale
    
    return (ub, lb, z)

#Sample uniformly from the line segment from lowerbound to upperbound
def sample(m, d, x_k, Vk, cons, scale=1):
    #m -- number of samples
    S = np.zeros((m, d)) #samples
    
    #x_k = normalize(x_k, [0] * d, d)
    
    ub, lb, direction = ub_lb(x_k, Vk, cons, scale)
    Sv = util.normalize(ub, lb, d)
    scale = util.distance(ub, lb)

    for i in range(m):
        var = random.uniform(0, scale) * Sv
        for j in range(d):
            S[i][j] = lb[j] + var[j]
    return S

def first_sample(m, d, x_k, cons, scale = 1):
    #m -- number of samples
    S = np.zeros((m, d)) #samples
    c = util.read_constraint(cons)
    
    #determin direction
    z = direction(d)
    
    #initialize upperbound and lowerbound
    ub = copy.deepcopy(x_k)
    lb = copy.deepcopy(x_k)

    #determin upper bound
    check_ub = True
    while check_ub :
        for i in range(d):
            if ub[i] > c[i][1]:
                check_ub = False
                break
            else:
                ub[i] += z[i] * scale
    
    for j in range(d):
        ub[j] -= z[j] * scale
    
    #determin lower bound
    check_lb = True
    while check_lb :
        for i in range(d):
            if lb[i] < c[i][0]:
                check_lb = False
                break
            else:
                lb[i] -= z[i] * scale
        
    for j in range(d):
        lb[j] += z[j] * scale

    dis = util.distance(ub, lb)

    for i in range(m):
        var = random.uniform(0, dis) * z
        for j in range(d):
            S[i][j] = lb[j] + var[j]
    
    return S
