import numpy as np

# Create a tridiag matrix 1-4-1 with 20 entries
def create_tridiag_matrix(n):
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        A[i, i] = 4.0
        if i > 0:
            A[i, i-1] = 1.0
        if i < n - 1:
            A[i, i+1] = 1.
    # System is periodic, so A(0, n-1) and A(n-1, 0) are also 1
    A[0, n-1] = 1.0
    A[n-1, 0] = 1.0
    return A

# Create rhs = 1:N
def create_rhs(n):
    return np.arange(1, n + 1, dtype=np.float32)

# Solve the system Ax = b using numpy's solver
def solve_system(A, b):
    return np.linalg.solve(A, b)

# Solve system and save to a .dat file
def save_solution_to_file(filename, x):
    np.savetxt(filename, x, fmt='%.6f')
    
if __name__ == "__main__":
    n = 200
    A = create_tridiag_matrix(n)
    b = create_rhs(n)
    x = solve_system(A, b)
    save_solution_to_file("unitt_32_sol.dat", x)