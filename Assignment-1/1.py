import numpy as np


def SimplexAlgorithm(A, B, C, X):
    while True:
        tgt = np.where(np.abs(A @ X - B) == 0)[0]
        A_tgt = A[tgt]
        utgt = np.where(np.abs(A @ X - B) != 0)[0]
        A_utgt = A[utgt]
        B_utgt = B[utgt]

        AI_col = -(np.linalg.inv(A_tgt)).T
        cost = AI_col @ C
        positive_cost_directions = np.where(cost > 0)[0]

        if len(positive_cost_directions) == 0:
            print('Optimal Solution :', X)
            print('Objective Value : ', C @ X)
            return X

        AI_col_c = AI_col[positive_cost_directions[0]]

        if len(np.where(np.dot(A, AI_col_c) > 0)[0]) == 0:
            print('Given LP is Unbounded')
            return None

        n = B_utgt - (A_utgt @ X)
        d = A_utgt @ AI_col_c
        n = n[np.where(d > 0)[0]]
        d = d[np.where(d > 0)[0]]
        s = n / d
        t = np.min(s[s >= 0])
        X = X + t * AI_col_c


if __name__ == "__main__":
    # A = np.array([[1, -1], [2, -1], [-1, 0], [0, -1]])
    # B = np.array([10, 40, 0, 0])
    # C = np.array([2, 1])
    # X = np.array([10, 0])
    A = np.array([[2, 1, 0], [1, 2, -2], [0, 1, 2], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    B = np.array([10, 20, 5, 0, 0, 0])
    C = np.array([2, -1, 2])
    X = np.array([5, 0, 0])
    SimplexAlgorithm(A, B, C, X)
