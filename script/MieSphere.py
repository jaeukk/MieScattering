import scipy.special as sp
import numpy as np

"""Utility functions for computing Mie coefficients and cross sections for a
single dielectric sphere."""

def spherical_hankel2(n,x):
    return sp.spherical_jn(n,x)-1j*sp.spherical_yn(n,x)

def psi(n,x):
    return x*sp.spherical_jn(n,x)
def zeta(n,x):
    return x*spherical_hankel2(n,x)

def dpsi(n,x):
    """Derivative of the Riccati–Bessel function ``psi(n, x)``."""
    return 0.5*(psi(n-1,x)-psi(n+1,x)+sp.spherical_jn(n,x))
def dzeta(n,x):
    """Derivative of the Riccati–Hankel function ``zeta(n, x)``."""
    return 0.5*(zeta(n-1,x)-zeta(n+1,x)+spherical_hankel2(n,x))

""" orientational """
def arr_pi(theta: float, n_max: int):
    """Compute ``pi_n`` terms (angular functions) up to order ``n_max``.

    Notes
    -----
    The expression divides by ``sin(theta)``, so callers should avoid values
    where ``theta`` is an integer multiple of ``pi`` to prevent numerical
    issues.
    """
    x = np.cos(theta)
    return np.array([sp.lpmv(1, n, x) for n in range(1,n_max)])/np.sin(theta)

def arr_tau(theta: float, n_max: int):
    """Compute ``tau_n`` terms (angular functions) up to order ``n_max``.

    The derivative of the associated Legendre polynomials is written
    explicitly to avoid relying on SciPy internals.
    """
    x = np.cos(theta)
    dPn = lambda n, val: (-(1 + n) * val * sp.lpmv(1, n, val) + n * sp.lpmv(1, 1 + n, val)) / (val**2 - 1)
    values = np.array([dPn(n, x) for n in range(1, n_max)])
    return -np.sin(theta) * values

def a_coeff(n, x, m):
    y = m*x
    numerator = dpsi(n,y) * psi(n,x) - m*psi(n,y) * dpsi(n,x)
    denominator = dpsi(n,y) * zeta(n,x) - m*psi(n,y) * dzeta(n,x)
    return numerator / denominator

def b_coeff(n, x, m):
    y = m*x
    numerator = m*dpsi(n,y) * psi(n,x) - psi(n,y) * dpsi(n,x)
    denominator = m*dpsi(n,y) * zeta(n,x) - psi(n,y) * dzeta(n,x)
    return numerator / denominator


def dsigma_ (ka, eps, thetas, M=50):
    """Dimensionless differential scattering cross section ``dσ/dΩ / a²``.

    Parameters
    ----------
    ka : float
        Dimensionless size parameter where ``k`` is the wavenumber in the
        background medium and ``a`` is the sphere radius.
    eps : float or complex
        Relative dielectric constant of the sphere with respect to the
        background.
    thetas : array_like
        Scattering angles in radians at which the differential cross section is
        evaluated.
    M : int, optional
        Number of series coefficients to include in the computation, by
        default 50.

    Returns
    -------
    ndarray
        Unpolarized differential scattering cross section at each requested
        angle. The two polarization amplitudes are combined following Bohren &
        Huffman: ``( |S₁|² + |S₂|² ) / (2 k² a²)``.
    """
    m = np.sqrt(eps)
    bn = np.array([ b_coeff(n,ka,m) for n in range(1,M)])
    an = np.array([ a_coeff(n,ka,m) for n in range(1,M)])

    # print(bn)
    # print(an)
    result = np.zeros((len(thetas),2))
    S = np.full((len(thetas),2), 0, dtype=np.complex128)
 
    for i, theta in enumerate(thetas):
        pi_n = arr_pi(theta, M) #np.array([pi_(n,theta) for n in range(1,M)])
        tau_n = arr_tau(theta, M) #np.array([tau(n,theta) for n in range(1,M)])

        S[i,0] = np.sum([ (2*n+1)/(n*(n+1)) * ( an[idx]*pi_n[idx] + bn[idx] * tau_n[idx] ) for idx,n in enumerate(range(1,M))])
        S[i,1] = np.sum([ (2*n+1)/(n*(n+1)) * ( bn[idx]*pi_n[idx] + an[idx] * tau_n[idx] ) for idx,n in enumerate(range(1,M))])
        
    result = (np.abs(S[:,0])**2 + np.abs(S[:,1])**2) / (2*ka**2)
    return result

def sigma_ (ka, eps, M=50):
    """Dimensionless total scattering cross section of a single sphere
    divided by the square of the sphere radius (``σ / a²``).

    Parameters
    ----------
    ka : float or array_like
        Dimensionless size parameter ``k · a`` where ``k`` is the wavenumber in
        the reference medium and ``a`` is the sphere radius.
    eps : float or complex
        Relative dielectric constant of the sphere with respect to the
        surrounding medium.
    M : int, optional
        Number of coefficients used in the Mie series, by default 50.

    Returns
    -------
    float or ndarray
        Total scattering cross section. Returns a scalar when ``ka`` is
        provided as a scalar, otherwise returns an array aligned with the input
        ``ka`` values.
    """
    m = np.sqrt(eps)
    input_scalar = np.isscalar(ka)
    ka_array = np.atleast_1d(ka)

    bn = np.array([b_coeff(n, ka_array, m) for n in range(1, M)])
    an = np.array([a_coeff(n, ka_array, m) for n in range(1, M)])

    result = np.inner(np.abs(an) ** 2 + np.abs(bn) ** 2, 2 * np.arange(1, M) + 1)
    result *= 2 * np.pi / ka_array**2

    if input_scalar:
        return result.item()
    return result
