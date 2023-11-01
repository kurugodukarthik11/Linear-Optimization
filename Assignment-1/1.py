import numpy as np


def simplex(A, B, C, X):
    while True:
        tgt = np.where(np.abs(A @ X - B) == 0)[0]
        A_tgt = A[tgt]
        utgt = np.where(np.abs(A @ X - B) != 0)[0]
        A_utgt = A[utgt]
        B_utgt = B[utgt]

        Z = -(np.linalg.inv(A_tgt)).T
        costs = Z @ C
        positive_cost_directions = np.where(costs > 0)[0]

        if len(positive_cost_directions) == 0:
            print('Optimal Solution :', X)
            print('Objective Value : ', C @ X)
            return X

        v = Z[positive_cost_directions[0]]

        if len(np.where(np.dot(A, v) > 0)[0]) == 0:
            print('Given LP is Unbounded')
            return None

        n = B_utgt - (A_utgt @ X)
        d = A_utgt @ v
        n = n[np.where(d > 0)[0]]
        d = d[np.where(d > 0)[0]]
        s = n / d
        t = np.min(s[s >= 0])
        X = X + t * v


if __name__ == "__main__":
    A = np.array([[1, -1], [2, -1], [-1, 0], [0, -1]])
    B = np.array([10, 40, 0, 0])
    C = np.array([2, 1])
    X = np.array([10, 0])
    simplex(A, B, C, X)
