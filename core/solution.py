import numpy as np

'''
Perform a step-by-step display
of the transformed matrix
'''


def solutions(matrix_a, matrix_b, show=True):
    try:
        A_mat = np.array(matrix_a, float).copy()
        b_mat = np.array(matrix_b, float).copy()
        length = len(b_mat)
    except ValueError as err:
        raise ValueError("Matrix must be numbers") from err

    if show:
        print(f"Initial Augmented Matrix [A | b]")
        print_augmented_matrix(A_mat, b_mat)

        steps = []

        # store initial state
        steps.append(
            {
                'step': 0,
                'desc': 'Initial matrix',
                'A': A_mat.copy(),
                'b_mat': b_mat.copy(),
            }
        )

        steps_counter = 1

        for k in range(length):
            if show:
                print(f"Step {steps_counter}: Pivot row {k + 1}")
                print(f"Pivot position: A[{k + 1},{k + 1}] (value = {A_mat[k, k]:.6f})")

            # partial pivoting (only when necessary)
            if np.fabs(A_mat[k, k]) < 1.0e-12:
                pivot_found = False
                for i in range(k + 1, length):
                    if np.fabs(A_mat[i, k]) > np.fabs(A_mat[k, k]):
                        if show:
                            print(f"Partial pivoting: Swap row {k + 1} with row {i + 1}")
                            print(f"Logic: |{A_mat[i, k]:.6f}| > |{A_mat[k, k]:.6f}|")

                        # swap rows in A
                        for j in range(length):
                            A_mat[k, j], A_mat[i, j] = A_mat[i, j], A_mat[k, j]
                        # swap rows in b
                        b_mat[k], b_mat[i] = b_mat[i], b_mat[k]

                        pivot_found = True

                        # Store step
                        steps.append(
                            {
                                'step': steps_counter,
                                'desc': f"Partial pivoting: R{k + 1} <-> R{i + 1}",
                                'A': A_mat.copy(),
                                'b_mat': b_mat.copy(),
                                'pivot_row': k + 1
                            }
                        )
                        steps_counter += 1

                        if show:
                            print("After pivoting")
                            print_augmented_matrix(A_mat, b_mat)
                        break

                if not pivot_found and show:
                    print("Pivot element is small\nNo better pivot found")

            # Normalize pivot row (pivot = 1)
            if show and A_mat[k, k] != 1:
                print(f"Normalize row {k + 1}: Divide by pivot {A_mat[k, k]:.6f}")

            pivot = A_mat[k, k]
            if pivot != 0:
                for j in range(k, length):
                    A_mat[k, j] /= pivot
                b_mat[k] /= pivot

            # Store step after normalization
            steps.append(
                {
                    'step': steps_counter,
                    'desc': f'Normalize R{k + 1}: R{k + 1} / {pivot:.6f}',
                    'A': A_mat.copy(),
                    'b_mat': b_mat.copy(),
                    'pivot_row': k + 1
                }
            )

            steps_counter += 1

            if show:
                print(f"After normalization:")
                print_augmented_matrix(A_mat, b_mat)

            # Elimination: make all other rows zero in the pivot column
            for i in range(length):
                if i == k or A_mat[i, k] == 0:
                    continue

                factor = A_mat[i, k]
                if show:
                    print(f"Eliminate from row {i + 1}: R{i + 1} <- R{i + 1} - ({factor:.6f}) * R{k + 1}")

                for j in range(k, length):
                    A_mat[i, j] -= factor * A_mat[k, j]
                b_mat[i] -= factor * b_mat[k]

                # Store elimination step
                steps.append(
                    {
                        'step': steps_counter,
                        'desc': f'Eliminate: R{i + 1} <- R{i + 1} - ({factor:.6f}) * R{k + 1}',
                        'A': A_mat.copy(),
                        'b_mat': b_mat.copy(),
                        'operation': f'R{i + 1} - {factor:.4f} * R{k + 1}'
                    }
                )
                steps_counter += 1

            if show and k < length - 1:
                print(f"Current matrix after elimination of column {k + 1}:")
                print_augmented_matrix(A_mat, b_mat)

        if show:
            print(f"Result:")
            # Echelon = Identity matrix
            print("Reduced Row Echelon Form")
            print_augmented_matrix(A_mat, b_mat)

            print("\nSolution Vector:")
            for i in range(length):
                print(f" x{i + 1} = {b_mat[i]:.6f}")

        return b_mat, A_mat, steps


def print_augmented_matrix(A, b):
    """
    Print augmented matrix in a readable format
    """
    n = len(b)
    for i in range(n):
        row_str = "["
        for j in range(n):
            row_str += f"{A[i, j]:10.6f}"
        row_str += f" | {b[i]:10.6f} ]"
        print(row_str)

def display_all_steps(steps):
    for each_step in steps:
        print(f"Step {each_step['step']}: {each_step['desc']}")
        print_augmented_matrix(each_step['A'], each_step['b_mat'])
        if 'pivot_row' in each_step:
            print(f"Pivot row {each_step['pivot_row']}")


def check_soln(origA, origB, solution):
    A = np.array(origA, float)
    b = np.array(origB, float)
    x = np.array(solution, float)

    print("Verifying solution")

    # calculate A*x
    calc_b = np.dot(A, x)

    print("Original Vector: ", b)
    print("Calculated A*x: ", calc_b)

    # calculate error
    error = np.abs(b - calc_b)
    print("Error (|b - A*x|): ", error)
    print("Maximum error: ", np.max(error))

    if np.all(error < 1e-10):
        print("Solution correct!")
    else:
        print("Solution NOT correct!")

    return error
