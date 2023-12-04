"""
Assignment 4
Karthik Kurugodu - MA20BTECH11008
Nikhil Kongara - MA20BTECH11011
Sai Prateek Katkam - ES20BTECH11024
Prajwaldeep Kamble - MA20BTECH11013
"""

import numpy as np
import sympy as sp
import csv

eps = 1e-6
stepsize = 1e-3


def read_input(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    C = np.array(data[0][:-1], dtype=float)
    B = np.array(data[1:], dtype=float)
    B = B[:, -1]
    A = np.array(data[1:], dtype=float)
    A = A[:, :-1]
    A = np.asarray(A)
    return C, B, A


def get_feasible_point(A, B, C):
    z = np.zeros(C.shape[0])
    if np.all(np.dot(A, z) <= B):
        return z
    A = np.hstack([A, np.ones(shape=(A.shape[0], 1))])
    z = np.hstack([z, np.min(B)])
    while len(np.where(np.abs(np.dot(A, z) - B) < eps)[0]) < A.shape[1]:
        tgt = np.where(np.abs(np.dot(A, z) - B) < eps)[0]
        A_tgt = A[tgt]
        A_tgt_ = sp.Matrix(A_tgt)
        N = A_tgt_.nullspace()
        if len(N) != 0:
            N = N[0]
            N = np.array(N).reshape(-1, 1)
            v = N
            v = v.reshape(-1,)
        else:
            break
        z = z + stepsize * v
    z = z.astype('float64')
    return z[:-1]


def check_degeneracy(A, B, C, X):
    if len(np.where(np.abs(np.dot(A, X) - B) < eps)[0]) == C.shape[0]:
        return False
    return True


def to_non_degenerate(A, B, C, X):
    deg_rows_n = A.shape[0] - C.shape[0]
    iter = 0
    while True:
        if iter < 1e4:
            iter += 1
            Bcopy = B
            Bcopy[:deg_rows_n] = Bcopy[:deg_rows_n] + (np.random.uniform(0, 1, size=deg_rows_n)) * eps*10
        else:
            Bcopy = B
            Bcopy[:deg_rows_n] = Bcopy[:deg_rows_n] + np.random.uniform(0.1, 10, size=deg_rows_n)
        if not check_degeneracy(A, Bcopy, C, X):
            print("Degeneracy Removed")
            break
    return Bcopy


def simplex(A, B, C, X):
    iteration_number = 1
    while True:
        tgt = np.where(np.abs(np.dot(A, X) - B) < eps)[0]
        A_tgt = A[tgt]
        utgt = np.where(np.abs(np.dot(A, X) - B) >= eps)[0]
        A_utgt = A[utgt]
        B_utgt = B[utgt]

        Z = -(np.linalg.inv(A_tgt)).T

        costs = np.dot(Z, C)
        positive_cost_directions = np.where(costs > 0)[0]

        if len(positive_cost_directions) == 0:
            print("****Optimal Solution Found****")
            print("")
            print("Iteration Number:", iteration_number)
            print('Optimal Solution :', X)
            print('Objective Value : ', np.dot(C, X))
            return X

        v = Z[positive_cost_directions[0]]

        # Unboundedness
        if len(np.where(np.dot(A, v) > 0)[0]) == 0:
            print("****Optimal Solution Not Found****")
            print("")
            print('Given LP is Unbounded')
            exit()

        print("Iteration Number:", iteration_number)
        print("Current Vertex : ", X)
        print("Current Objective Value : ", np.dot(C, X))
        print("")

        indices = np.where(np.dot(A_utgt, v) > 0)[0]
        slack_variables = (B_utgt - (np.dot(A_utgt, X)))[indices]
        step_size = (np.dot(A_utgt, v))[indices]
        s = slack_variables / step_size
        t = np.min(s[s >= 0])
        X = X + t * v
        iteration_number += 1
    return X


def SimplexAlgorithm(A, B, C):
    X = get_feasible_point(A, B, C)
    B = to_non_degenerate(A, B, C, X)
    X = np.round(X, decimals=int(np.abs(np.floor(np.log10(eps)))))
    X = simplex(A, B, C, X)
    return X


if __name__ == "__main__":
    C, B, A = read_input(file_path='test_cases/4.1.csv')
    OptimalSolution = SimplexAlgorithm(A, B, C)
