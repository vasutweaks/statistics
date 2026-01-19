import numpy as np


def genweights(p, q, dt=1):
    """
    Calculates the optimal weighting coefficients (cpn) and noise reduction factor.

    Args:
        p: Number of points before the point of interest (negative).
        q: Number of points after the point of interest (positive).
        dt: Sampling period (defaults to 1).

    Returns:
        cn: Vector of optimal weighting coefficients.
        error: Noise reduction factor.

    Raises:
        ValueError: If p is not less than q or incorrect number of arguments provided.
    """

    # if nargin < 2:
    # raise ValueError('Incorrect number of arguments.')
    # elif nargin < 3:
    # dt = 1

    # Verification
    p = max(p, -p)
    q = max(q, -q)
    print(p, q)
    # if p > q:
    # raise ValueError('p must be less than q')

    if not isinstance(p, int) or not isinstance(q, int):
        raise ValueError("p and q must be integers")
    # if p >= 0 or q <= 0:
    # raise ValueError("p must be negative and q must be positive")
    # if p > -q:
    # raise ValueError("p must be less than -q")
    # Total number of coefficients and matrix size
    # Build matrices
    N = abs(p) + abs(q)
    T = N + 1
    A = np.zeros((T, T))
    A[T - 1, :] = np.concatenate([np.ones(N), [0]])
    n = np.arange(-p, q + 1)
    print(n)
    n = n[n != 0]
    print(n)
    for i, val in enumerate(n):
        A[i, :] = np.array([-val / (2 * n[i]), val**2 * dt**2 / 4])
        A[i, i] = -1

    B = np.zeros((T, 1))
    B[-1] = 1

    # Compute coefficients and error
    cn = np.linalg.solve(A, B)[:-1]
    error = np.sqrt(np.sum(cn**2 / (n * dt)) ** 2 + np.sum((cn**2 / (n * dt)) ** 2))

    return cn, error


# Example usage
p = -6
q = 5
dt = 0.5
cn, error = genweights(p, q, dt)

print(f"Optimal weighting coefficients: {cn}")
print(f"Noise reduction factor: {error}")
