"""
Assignment 2
Karthik Kurugodu - MA20BTECH11008
Nikhil Kongara - MA20BTECH11011
Sai Prateek Katkam - ES20BTECH11024
Prajwaldeep Kamble - MA20BTECH11013
"""

import numpy as np
import sympy as sp
import csv

eps = 1e-4
stepsize = 1e-3


def read_input(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    X = np.array(data[0][:-1], dtype=float)
    C = np.array(data[1][:-1], dtype=float)
    B = np.array(data[2:], dtype=float)
    B = B[:, -1]
    A = np.array(data[2:], dtype=float)
    A = A[:, :-1]
    A = np.asarray(A)
    return X, C, B, A


def check_feasibility(A, B, C, X):
    if len(np.where(np.dot(A, X) - B > eps)[0]) != 0:
        print("Given point is not feasible")
        return False
    return True


def to_near_boundary(A, B, C, X):
    if len(np.where(np.abs(np.dot(A, X) - B) < eps)[0]) == 0:
        v = np.ones((A.shape[1], 1))
        while not len(np.where(np.abs((np.dot(A, X)) - B) < eps)[0]) >= A.shape[1]:
            temp = X - stepsize * v
            if len(np.where(np.dot(A, X) > B)[0]) == 0:
                X = temp
                X = X.reshape(-1, 1)
            else:
                break
        print(f"Nearest point on Boundary : {X.reshape(-1, )}")
    else:
        print(f"X: {X.reshape(-1, )} is already on boundary")
    X = X.astype('float64')
    return X


def to_vertex(A, B, C, X):
    if len(np.where(np.abs((np.dot(A, X)) - B) < eps)[0]) < A.shape[1]:
        while len(np.where(np.abs((np.dot(A, X)) - B) < eps)[0]) < A.shape[1]:
            tgt = np.where(np.abs(np.dot(A, X) - B) < eps)[0]
            A_tgt = A[tgt]
            A_tgt_ = sp.Matrix(A_tgt)
            N = A_tgt_.nullspace()
            if len(N) != 0:
                N = N[0]
                N = np.array(N).reshape(-1, 1)
                v = N
            else:
                break
            X = X - stepsize * v
        X = X.astype('float64')
        print(f"Vertex : {X.reshape(-1, )}\n")
    else:
        print(f"X: {X.reshape(-1, )} is already a vertex\n")
    return X


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

        if len(np.where(np.dot(A, v) > 0)[0]) == 0:
            print("****Optimal Solution Not Found****")
            print("")
            print('Given LP is Unbounded')
            return None

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


def SimplexAlgorithm(X, C, B, A):
    B, C, X = B.reshape(-1, 1), C.reshape(-1, 1), X.reshape(-1, 1)
    if not check_feasibility(A, B, C, X):
        print("Given point is not feasible")
        return None
    print("Initial Feasible point : ", X.reshape(-1, ))
    X = to_near_boundary(A, B, C, X)
    X = to_vertex(A, B, C, X)
    B, C, X = B.reshape((-1,)), C.reshape((-1,)), X.reshape((-1,))
    X = np.round(X, decimals=int(np.abs(np.floor(np.log10(eps)))))
    X = simplex(A, B, C, X)
    return X


if __name__ == "__main__":
    X, C, B, A = read_input(file_path="test_cases/2.3.csv")
    OptimalSolution = SimplexAlgorithm(X, C, B, A)
