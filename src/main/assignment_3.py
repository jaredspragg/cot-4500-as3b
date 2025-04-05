import numpy as np

# 1. Gaussian Elimination

def gaussian_elimination():
    A = np.array([[2.0, -1.0, 1.0], [1.0, 3.0, 1.0], [-1.0, 5.0, 4.0]])
    b = np.array([6.0, 0.0, -3.0])
    n = len(b)
    
    # Forward elimination
    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]

    return x

# 2. LU Factorization

def lu_factorization():
    A = np.array([
        [1.0, 1.0, 0.0, 3.0],
        [2.0, 1.0, -1.0, 1.0],
        [3.0, -1.0, -1.0, 2.0],
        [-1.0, 2.0, 3.0, -1.0]
    ])
    n = len(A)
    L = np.eye(n)
    U = A.copy()

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            U[j] = U[j] - factor * U[i]

    determinant = np.linalg.det(A)
    return determinant, L, U

# 3. Diagonally Dominant Check

def is_diagonally_dominant():
    A = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ])
    for i in range(len(A)):
        row_sum = sum(abs(A[i][j]) for j in range(len(A)) if j != i)
        if abs(A[i][i]) < row_sum:
            return False
    return True

# 4. Positive Definite Check

def is_positive_definite():
    A = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ])
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Run all parts
if __name__ == "__main__":
    # Euler and RK for comparison
    from math import pow

    def func(t, y):
        return t - y**2

    def euler(t0, y0, t_end, steps):
        h = (t_end - t0) / steps
        t, y = t0, y0
        for _ in range(steps):
            y += h * func(t, y)
            t += h
        return y

    def rk4(t0, y0, t_end, steps):
        h = (t_end - t0) / steps
        t, y = t0, y0
        for _ in range(steps):
            k1 = h * func(t, y)
            k2 = h * func(t + h / 2, y + k1 / 2)
            k3 = h * func(t + h / 2, y + k2 / 2)
            k4 = h * func(t + h, y + k3)
            y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t += h
        return y

    print(euler(0, 1, 2, 10))
    print(rk4(0, 1, 2, 10))

    print(gaussian_elimination())

    determinant, L, U = lu_factorization()
    print(determinant)
    print(L)
    print(U)

    print(is_diagonally_dominant())
    print(is_positive_definite())

 
