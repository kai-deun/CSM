# TODO: add comm
def square_matrix():
    tooltip = "at least 2x2"
    while True:
        try:
            num = int(input(f"Enter the matrix size {tooltip}: "))
            if num >= 2:
                break
            else:
                print(f"Please enter the matrix size {tooltip}.")
        except ValueError:
            print("Invalid input, use numerics only.")

    print(f"Enter the {num}x{num} matrix A, row by row")
    A = [] # initialized empty arrays
    for i in range(num):
        while True:
            row = input(f"Enter the row {i+1}: ")
            try:
                row = [float(x) for x in row.split()]
                if len(row) != num:
                    print(f"Please enter the exact {num} cels.")
                    continue
                A.append(row)
                break
            except ValueError:
                print("Invalid input, use numerics only.")
    return A, num

# TODO: add comm
def vector(num):
    print(f"Enter vector (index {num})")
    while True:
        str = input("b: ")
        try:
            b = [float(x) for x in str.split()]
            if len(b) != num:
                print(f"Please enter the exact {num}x{num} cel.")
                continue
            return b
        except ValueError:
            print("Invalid input, use numerics only.")