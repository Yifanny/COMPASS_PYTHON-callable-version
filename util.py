import math
import numpy as np
import numpy.matlib as npm
import random
import copy

def distance(v1, v2):
    d = [0] * len(v1)
    result = 0
    for i in range(len(v1)):
        d[i] = v1[i] - v2[i]
        result += d[i] ** 2
    return math.sqrt(result)

def normalize(ub, lb, d):
    v = np.random.normal(size = d)
    for i in range(d):
        v[i] = ub[i] - lb[i]
    norm = distance(ub, lb)
    #print(norm)
    if norm == 0:
        print(ub)
        print(lb)
    return 1/norm * v
    
def remove(mat, index):
    return np.delete(mat, index, 0)

def mat_sub(mat1, mat2):
    x1, y1 = mat1.shape
    x2, y2 = mat2.shape
    
    if x1 == x2 and y1 == y2:
        result = np.zeros((x1, y1))
        for i in range(x1):
            for j in range(y1):
                result[i][j] = mat1[i][j] - mat2[i][j]
    else:
        print('please check the shape of two matrix.')
    return result

def mat_add(mat1, mat2):
    x1, y1 = mat1.shape
    x2, y2 = mat2.shape
    
    if x1 == x2 and y1 == y2:
        result = np.zeros((x1, y1))
        for i in range(x1):
            for j in range(y1):
                result[i][j] = mat1[i][j] + mat2[i][j]
    else:
        print('please check the shape of two matrix.')
    return result

def mat_abs(mat):
    m, d = mat.shape
    for i in range(m):
        for j in range(d):
            mat[i][j] = abs(mat[i][j])
    return mat

def read_constraint(c):
    result = np.zeros((len(c), 2))
    for i in range(len(c)):
        for j in range(2):
            result[i][j] = c[i][j]
    return result
   
def ismember(row, mat):
    m, n = mat.shape
    find = False
    while not find:
        for i in range(m):
            for j in range(n):
                if row[j] != mat[i][j]:
                    break
                elif j == n-1:
                    index = i
                    find = True
    return index