import numpy as np


'''
The algorithm is originally and credited to "Murad Elarbi"
Others such as gui and implementation of user inputs are made
by Iris
'''

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
    # sample TODO: remove later and implement unit testing
    a = [[12, 2, 0, 1],
         [2, 0, 3, 2],
         [4, -3, 0, 1],
         [6, 1, -6, -5]]

    b = [0, -2, -7, 6]

    X, A = gaussjordan(a, b)

    print(f"Solution \n {X}")
    print(f"Transformed Matrix[A] \n {A}")

# TODO: implement user input
if __name__ == "__main__":
    main()
