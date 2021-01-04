# Code Generating Discrete Legendre Orthogonal Polynomials and the
# Legendre Delay Network Basis
#
# Andreas Stöckel, December 2020
#
# The code in this file is licensed under the Creative Commons Zero license.
# To the extent possible under law, Andreas Stöckel has waived all copyright and
# related or neighboring rights to this code. You should have received a
# complete copy of the license along with this code. If not, please visit
#
#    https://creativecommons.org/publicdomain/zero/1.0/legalcode-plain.
#
# This work is published from: Canada.

import numpy as np
import scipy.linalg

## Legendre Delay Network (LDN)


def mk_ldn_lti(q, dtype=np.float):
    """
    Generates the A, B matrices of the linear time-invariant (LTI) system
    underlying the LDN. 

    The returned A is a q x q matrix, the returned B is a vector of length q.
    Divide the returned matrices by the desired window length theta.

    See Aaron R. Voelker's PhD thesis for more information:
    https://hdl.handle.net/10012/14625 (Section 6.1.3, p. 134)
    """
    qs = np.arange(q)
    A = -np.ones((q, q), dtype=dtype)
    for d in range(1, q, 2):  # iterate over odd diagonals
        A[range(d, q), range(0, q - d)] = 1
    B = np.ones((q, ), dtype=dtype)
    B[1::2] = -1
    return (2 * qs[:, None] + 1) * A, \
           (2 * qs + 1) * B


def discretize_lti(dt, A, B):
    """
    Discretizes an LTI system described by matrices A, B under a
    zero-order-hold (ZOH) assumption. The new matrices Ad, Bd can be used in
    the following manner

       x[t + 1] = Ad x[t] + Bd u[t] ,

    i.e., the returned matrices implicitly contain the
    integration step.

    See https://en.wikipedia.org/wiki/Discretization for
    more information.
    """
    # See https://en.wikipedia.org/wiki/Discretization
    Ad = scipy.linalg.expm(A * dt)
    Bd = np.linalg.solve(A, (Ad - np.eye(A.shape[0])) @ B)
    return Ad, Bd


## Legendre Delay Network Basis


def mk_ldn_basis_naive(q, N=None, normalize=True):
    """
    This function is the attempt at generating a LDN basis using naive Euler
    integration. This produces horribly wrong results.
    
    For reference only, DO NOT USE. Use `mk_ldn_basis` instead.
    """
    q, N = int(q), int(q) if N is None else int(N)
    A, B = mk_ldn_lti(q)
    At, Bt = A / N + np.eye(q), B / N
    res = np.zeros((q, N))
    Aexp = np.eye(q)
    for i in range(N):
        res[:, q - i - 1] = Aexp @ Bt
        Aexp = At @ Aexp
    return (res / np.linalg.norm(res, axis=1)[:, None]) if normalize else res


def mk_ldn_basis(q, N=None, normalize=True):
    """
    Generates the LDN basis for q basis vectors and N input samples.  Set
    `normalize` to `False` to obtain the exact LDN impulse response, otherwise
    a normalized basis transformation matrix as defined in the TR is returned.
    """
    q, N = int(q), int(q) if N is None else int(N)
    At, Bt = discretize_lti(1.0 / N, *mk_ldn_lti(q))
    res = np.zeros((q, N))
    Aexp = np.eye(q)
    for i in range(N):
        res[:, N - i - 1] = Aexp @ Bt
        Aexp = At @ Aexp
    return (res / np.linalg.norm(res, axis=1)[:, None]) if normalize else res


## Discrete Legendre Orthogonal Polynomial Basis and Related Code


def mk_leg_basis(q, N=None):
    """
    Creates a non-orthogonal basis by simply sampling the Legendre polynomials.
    """
    q, N = int(q), int(q) if N is None else int(N)
    xs0 = np.linspace(0.0, 1.0, N + 1)[:-1]
    xs1 = np.linspace(0.0, 1.0, N + 1)[1:]
    res = np.zeros((q, N))
    for n in range(q):
        Pn = np.polynomial.Legendre([0] * n + [1], [1, 0]).integ()
        res[n] = Pn(xs1) - Pn(xs0)
    return res / np.linalg.norm(res, axis=1)[:, None]


def mk_dlop_basis_linsys(q, N=None):
    """
    Constructs a matrix of "Discrete Legendre Orthogonal Polynomials" (DLOPs).
    q is the number of polynomials to generate, N is the number of samples for
    each polynomial.

    This is function is for reference only and should not be used. It is
    unstable for q > 30 (if N > q the function may be stable longer).

    This function uses a rather inefficient approach that directly relies on
    the definition of a Legendre Polynomial (a set of orthogonal Polynomials
    with Pi(1) = 1.0) to generate the basis.

    In each iteration i, this function adds a new polynomial of degree i to the
    set of already computed polynomials. The polynomial coefficients are
    determined by solving for coefficients that generate discrete sample points
    that are orthogonal to the already sampled basis vectors.

    The returned basis is made orthogonal by dividing by the norm of each
    discrete polynomial.
    """
    # Construct the sample points
    q, N = int(q), int(q) if N is None else int(N)
    qs, Ns = np.arange(q), np.arange(N)
    xs = 2.0 * Ns / (N - 1.0) - 1.0

    # Evaluate the individual monomials (this is a Vandermonde matrix)
    M = np.power(xs[:, None], qs[None, :])

    # Create the matrix. The first basis vector is "all ones"
    res = np.zeros((q, N))
    res[0] = 1.0

    # Solve for polynomial coefficients up to degree q such that the newly
    # added basis vector is orthogonal to the already created basis vectors,
    # and such that the last sample is one.
    for i in range(1, q):
        A = np.zeros((i + 1, i + 1))
        b = np.zeros((i + 1, ))
        b[-1] = 1.0
        A[:i, :] = res[:i, :] @ M[:, :i + 1]
        A[i, :] = M[0, :i + 1]
        coeffs = np.linalg.solve(A, b)
        res[i] = M[:, :i + 1] @ coeffs

    return res / np.linalg.norm(res, axis=1)[:, None]


def mk_dlop_basis_direct(q, N=None):
    """
    Slow, direct implementation of the DLOP basis according to

    Neuman, C. P., & Schonbach, D. I. (1974).
    Discrete (legendre) orthogonal polynomials—A survey.
    International Journal for Numerical Methods in
    Engineering, 8(4), 743–770.
    https://doi.org/10.1002/nme.1620080406

    Note that this code relies on the fact that Python 3 always uses
    "big ints" or ("long" in Python 2 terms). The integers used in this
    function will likely not fit into 32- or 64-bit integers; so be careful
    when porting this code to a different programing language.
    """
    def fading_factorial(K, m):
        # Fading factorial as defined in the paper.
        res = 1
        for i in range(m):
            res *= K - i
        return res

    def nCr(n, r):
        # Binomial coefficient (n choose r; nCr is what my trusty pocket
        # calculator calls it).
        return fading_factorial(n, r) // \
               fading_factorial(r, r)

    q, N = int(q), int(q) if N is None else int(N)
    res = np.zeros((q, N))
    for m in range(q):
        # Usa a common denominator instead of dividing by
        # fading_factorial(N - 1, j), where "j" is the inner loop variable.
        # Instead we divide all terms by fading_factorial(N - 1, m) and
        # multiply the terms by the additional terms that we're dividing by.
        # This way we can perform the final division numer / denom computing
        # the float output at the very end; everything up to this point is
        # precise integer arithmetic.
        denom = fading_factorial(N - 1, m)
        for K in range(N):
            numer = 0
            for j in range(m + 1):
                # Main equation from the paper. The last term corrects for the
                # common denominator.
                c = nCr(m, j) * nCr(m + j, j) * \
                   fading_factorial(K, j) * \
                   fading_factorial(N - 1 - j, m - j)
                numer += c if (j % 2 == 0) else -c
            res[m, K] = numer / denom
        res[m]

    return res / np.linalg.norm(res, axis=1)[:, None]


def mk_dlop_basis_recurrence(q, N=None):
    """
    Computes the DLOP basis using the Legendre recurrence relation as described
    in the section "Generation Scheme" of Neuman & Schonbach, 1974, pp. 758-759
    (see above for the full reference).

    Do NOT use this function. This function is numerically unstable and only
    included as a reference. Use `mk_dlop_basis` instead
    """

    # Fill the first rows
    q, N = int(q), int(q) if N is None else int(N)
    res = np.zeros((q, N))
    if q > 0:
        res[0] = np.ones(N)
    if q > 1:
        res[1] = np.linspace(1, -1, N)

    # Iterate over all columns
    for K in range(N):
        # Compute the initial coefficients for the recurrence relation
        c0, c1, c2 = 0, N - 2 * K - 1, N - 1
        δ0, δ1, δ2 = N - 1, 2 * c1, N - 1

        # Iterate over all rows
        for m in range(2, q):
            δ0, δ1, δ2 = δ0 + 2, δ1, δ2 - 2
            c0, c1, c2 = c0 + δ0, c1 + δ1, c2 + δ2
            res[m, K] = (c1 * res[m - 1, K] - c0 * res[m - 2, K]) / c2

    return res / np.linalg.norm(res, axis=1)[:, None]


def mk_dlop_basis(q, N=None, eps=1e-7):
    """
    Same as `mk_dlop_basis_recurrence`, but updates all columns at once using
    numpy.
    """

    # Fill the first rows
    q, N = int(q), int(q) if N is None else int(N)
    res = np.zeros((q, N))
    if q > 0:
        res[0] = np.ones(N) / np.sqrt(N)
    if q > 1:
        res[1] = np.linspace(1, -1, N) * np.sqrt((3 * (N - 1)) / (N * (N + 1)))

    # Pre-compute the coefficients c0, c1. See Section 4.4 of the TR.
    Ks = np.arange(0, N, dtype=np.float)[None, :]
    ms = np.arange(2, q, dtype=np.float)[:, None]
    α1s = np.sqrt(  ((2 * ms + 1) * (N - ms)) \
                  / ((2 * ms - 1) * (N + ms)))
    α2s = np.sqrt(  ((2 * ms + 1) * (N - ms) * (N - ms + 1)) \
                  / ((2 * ms - 3) * (N + ms) * (N + ms - 1)))
    β1s = α1s * ((2 * ms - 1) * (N - 2 * Ks - 1) / (ms * (N - ms)))
    β2s = α2s * ((ms - 1) * (N + ms - 1) / (ms * (N - ms)))

    # The mask is used to mask out columns that cannot become greater than one
    # again. This prevents numerical instability.
    mask = np.ones((q, N), dtype=np.bool)

    # Evaluate the recurrence relation
    for m in range(2, q):
        # A column K can only become greater than zero, if one of the
        # cells in the two previous rows was significantly greater than zero.
        mask[m] = np.logical_or(mask[m - 1], mask[m - 2])

        # Apply the recurrence relation
        res[m] = (  (β1s[m - 2]) * res[m - 1] \
                  - (β2s[m - 2]) * res[m - 2]) * mask[m]

        # Mask out cells that are smaller than some epsilon
        mask[m] = np.abs(res[m]) > eps

    return res


## Fourier and Cosine Basis


def mk_fourier_basis(q, N=None):
    """
    Generates the q x N matrix F that can be used to compute a Fourier-like
    transformation of a real-valued input vector of length N.  The first
    result dimension will be the DC offset.  Even result dimensions are the
    real (sine) Fourier coefficients, odd dimensions are the imaginary (cosine)
    coefficients.
    
    Beware that this is a only a Fourier transformation for q = N, and even
    then not a "proper" Fourier transformation because the transformation
    matrix is normalized to be orthogonal.  So be careful when comparing the
    results of this function to "standard" Fourier transformations.
    """
    q, N = int(q), int(q) if N is None else int(N)
    qs, Ns = np.arange(q)[:, None], np.arange(N)[None, :]
    freq = ((qs + 1) // 2)  # 0, 1, 1, 2, 2, ...
    phase = (qs % 2)  # 0, 1, 0, 1, 0, ...
    F = np.cos(
        2.0 * np.pi * freq * (Ns + 0.5) / N + \
        0.5 * np.pi * phase)
    F[0] /= np.sqrt(2)
    F[-1] /= np.sqrt(2) if (q % 2 == 0 and N == q) else 1.0
    return F * np.sqrt(2 / N)


def mk_cosine_basis(q, N=None):
    """
    Generates the q x N matrix C which can be used to compute the orthogonal
    DCT-II, everyone's favourite basis transformation.  As with the
    `mk_fourier_basis` function above, this code only returns a canonical
    DCT basis if q = N.
    """
    q, N = int(q), int(q) if N is None else int(N)
    qs, Ns = np.arange(q)[:, None], np.arange(N)[None, :]
    C = np.cos((Ns + 0.5) / N * qs * np.pi)
    C[0] /= np.sqrt(2)
    return C * np.sqrt(2 / N)


## Low-pass filtered bases


def lowpass_filter_basis(T, ratio=1.0):
    """
    Takes a basis T with shape q x N and returns a basis that additionally
    applies a low-pass filter to the N-dimensional input, such that the input
    is represented by (ratio * q) Fourier coefficients.
    
    This function has no effect if (ratio * q) = N; in this case, there is no
    potential for information loss.
    """
    q, N = T.shape
    F = mk_fourier_basis(int(ratio * q), N)
    return T @ F.T @ F

