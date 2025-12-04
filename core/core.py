import numpy as np
from user_input import *
from solution import *


'''
The algorithm is originally and credited to "Murad Elarbi"
Others such as gui and implementation of user inputs are made
by Iris

... add more documentation about the logic ...
'''

# MAIN ALGO
def gaussjordan(matrix_a, matrix_b):
    try:
        matrix_a = np.array(matrix_a, float)
        matrix_b = np.array(matrix_b, float)
        length = len(matrix_b)
    except ValueError as err:
        raise ValueError("Numbers only") from err

    # main loop from the algorithm
    for k in range(length):
        # partial pivoting
        if np.fabs(matrix_a[k, k]) < 1.0e-12:
            for i in range(k + 1, length):
                if np.fabs(matrix_a[i, k]) > np.fabs(matrix_a[k, k]):
                    for j in range(k, length):
                        matrix_a[k, j], matrix_a[i, j] = matrix_a[i, j], matrix_a[k, j]
                    matrix_b[k], matrix_b[i] = matrix_b[i], matrix_b[k]
                    break
        # division of the pivot row
        pivot = matrix_a[k, k]
        for j in range(k, length):
            matrix_a[k, j] /= pivot
        matrix_b[k] /= pivot
        # elimination loop
        for i in range(length):
            if i == k or matrix_a[i, k] == 0: continue
            factor = matrix_a[i, k]
            for j in range(k, length):
                matrix_a[i, j] -= factor * matrix_a[k, j]
            matrix_b[i] -= factor * matrix_b[k]
    return matrix_b, matrix_a

def main():
    A, n  = square_matrix()
    b = vector(n)

    # original gauss jordan function
    X, A_mat = gaussjordan(A, b)

    print("Solution:")
    print(" ".join(f"{val:.4f}" for val in X))

    print(A_mat)

    # for step-by-step display
    print("\nStep-by-step")
    X_steps, A_mat_steps, steps = solutions(A, b, show=True)

    # verify both solution
    print(f"Original: {X}")
    print(f"By steps: {X_steps}")

    if np.allclose(X, X_steps, rtol=1e-8):
        print("Solution correct!")
    else:
        print("Solution differ in numerical precision")

    # Verify if solution correct
    print("\nSolution verification")
    check_soln(A, b, X)

if __name__ == "__main__":
    main()
