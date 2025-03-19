import numpy as np

def generate_matrix(size, low=0, high=9):
    return np.random.randint(low, high + 1, (size, size))

def split_matrix(A):
    n = A.shape[0] // 2
    return A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]

def strassen_recursive(A, B, level=0):
    n = A.shape[0]
    if n <= 2:
        return A @ B
    
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    S1 = B12 - B22
    S2 = A11 + A12
    S3 = A21 + A22
    S4 = B21 - B11
    S5 = A11 + A22
    S6 = B11 + B22
    S7 = A12 - A22
    S8 = B21 + B22
    S9 = A11 - A21
    S10 = B11 + B12
    
    print(f"Level {level} S-matrices:")
    for i, S in enumerate([S1, S2, S3, S4, S5, S6, S7, S8, S9, S10], 1):
        print(f"S{i}:\n{S}\n")
    
    T1 = A11 + A22
    T2 = B11 + B22
    
    print(f"Level {level} T-matrices:")
    print(f"T1:\n{T1}\n")
    print(f"T2:\n{T2}\n")
    
    Q1 = strassen_recursive(A11, S1, level+1)
    Q2 = strassen_recursive(S2, B22, level+1)
    Q3 = strassen_recursive(S3, B11, level+1)
    Q4 = strassen_recursive(A22, S4, level+1)
    Q5 = strassen_recursive(T1, T2, level+1)
    Q6 = strassen_recursive(S7, S8, level+1)
    Q7 = strassen_recursive(S9, S10, level+1)
    
    print(f"Level {level} Q-matrices:")
    for i, Q in enumerate([Q1, Q2, Q3, Q4, Q5, Q6, Q7], 1):
        print(f"Q{i}:\n{Q}\n")
    
    C11 = Q5 + Q4 - Q2 + Q6
    C12 = Q1 + Q2
    C21 = Q3 + Q4
    C22 = Q5 + Q1 - Q3 - Q7
    
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C

np.random.seed(42)
A = generate_matrix(8)
B = generate_matrix(8)

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

C = strassen_recursive(A, B)
print("\nResulting Matrix C:")
print(C)
