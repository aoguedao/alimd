import numpy as np
from numba import jit


def duplication_matrix(n):
    dup = np.zeros((n ** 2, n))
    dup[np.arange(1, n + 1) * (n + 1) - n - 1, np.arange(n)] = 1
    return dup


def raw_dup(A):
    n2, _ = A.shape
    n = int(np.sqrt(n2))
    dup = duplication_matrix(n)
    return dup.T @ A @dup


def duplication_mul(A):
    n2, _ = A.shape
    n = int(np.sqrt(n2))
    return A[::n+1,::n+1]


@jit(nopython=True)
def duplication_mul_nopython(A):
    n2, _ = A.shape
    n = int(np.sqrt(n2))
    return A[::n+1,::n+1]


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    import sys
    args = sys.argv
    n = int(args[1])
    n2 = n ** 2
    A = np.random.randint(1, n2, size=(n2, n2))  # Crear matriz random
    print(f"A shape: {A.shape}")

    dup = duplication_matrix(n)
    first_second_bool = np.array_equal(dup.T @ A @dup, duplication_mul(A))
    second_third_bool = np.array_equal(duplication_mul(A), duplication_mul_nopython(A))
    print(f"Equals? --> {first_second_bool and second_third_bool}")

    print("#########")
    print("Benchmark")
    print("#########\n")
    import timeit
    print(f"Raw multiplication: {timeit.timeit('raw_dup(A)', globals=globals())}")
    print(f"Indexing multiplication: {timeit.timeit('duplication_mul(A)', globals=globals())}")
    print(f"Indexing numba multiplication: {timeit.timeit('duplication_mul_nopython(A)', globals=globals())}")



