import numpy as np
import sys
import random
import copy

try: import simplejson as json
except ImportError: import json

from collections import OrderedDict

import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
from mpl_toolkits.mplot3d import Axes3D
import time

import sample
import example
'''
def get_options():
    try:
        with open('config.json', 'r') as f:
            options = json.load(f)
    except:
        raise Exception("config.json did not load properly. Perhaps a spurious comma?")
    
    func = options["function-name"]
    conds = options["variables"]
    dim = len(conds)
    c = []
    for i in range(dim):
        c.append((conds[list(conds.keys())[i]]["min"], conds[list(conds.keys())[i]]["max"], conds[list(conds.keys())[i]]["type"]))
    s = options["num-samples"]
    iters = options["max-iters"]
    scale = options["scale"]
    
    return (func, dim, s, iters, c, scale)
'''

def showimage(result, iters):
    iters = int(iters)
    print(min(result))
    fig = plt.figure()
    plt.plot(range(iters), result)
    plt.show()

def compass(func, d, s, iters, c, scale):
    d = int(d)
    iters = int(iters)
    s = int(s)
    
    #initialization
    x_0 = [0] * d
    for i in range(d):
        if c[i][2] == 'int':
            x_0[i] = random.randint(c[i][0],c[i][1])
        elif c[i][2] == 'float':
            x_0[i] = random.uniform(c[i][0],c[i][1])
        else:
            print('please check the type of the contraint.')
    x_star = copy.deepcopy(x_0)
    Vk = np.array([x_0])
    result = [0] * iters
    
    #first sampling
    S = sample.first_sample(s, d, x_0, c, scale = 0.5)
    Vk = np.vstack((Vk, S))
    print('Vk after first sampling:')
    print(Vk)
    #tasks = [(x, y) for x in Vk[:,0] for y in Vk[:,1]]
    start = time.clock()
    pool = Pool(6)
    #Fx = pool.map(mnist, tasks)
    Fx = pool.map(func, Vk)
    pool.close()
    pool.join()
    end = time.clock()
    print('time:')
    print(end-start)
    print(Fx)
    x_index = Fx.index(min(Fx))
    x_star = copy.deepcopy(Vk[x_index])
    result[0] = min(Fx)
    
    #start iteration
    pool = Pool(5)
    for k in range(1, iters, 1):
        print(k)
        S = sample.sample(s, d, x_star, Vk, c, 0.5)
        Vk = np.vstack((Vk, S))
        print('Vk shape:')
        print(Vk.shape)
        start = time.clock()
        Fx_s = pool.map(func, S)
        end = time.clock() 
        print('time:')
        print(end-start)
        Fx.extend(Fx_s)
        x_index = Fx.index(min(Fx))
        x_star = Vk[x_index]
        result[k] = min(Fx)
        print('x_star:')
        print(x_star)
        print('min(Fx):')
        print(min(Fx))
    pool.close()
    pool.join()
    
    return (x_star, min(result), result)


def main():
    print('Please call this funciton in your python programme.')


if __name__ == '__main__':
    main()
