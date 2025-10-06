import scipy.special as sp
import numpy as np

def dJ(n,x):
    return 0.5*(sp.jv(n-1,x)-sp.jv(n+1,x))
def dH(n,x):
    return 0.5*(sp.hankel2(n-1,x)-sp.hankel2(n+1,x))
def H(n,x):
    return sp.hankel2(n,x)
def J(n,x):
    return sp.jv(n,x)

def a_coeff(n, x, m, first_order=False):
    """
    Computes the Mie scattering coefficient 'a_n' (associated with TE) for a cylindrical particle.

    Parameters
    ----------
    n : int
        Order (mode number).
    x : float
        Size parameter, defined as k * a (where k is the wavenumber and a is the cylinder radius).
    m : float 
        Relative refractive index of the cylinder (particle refractive index divided by medium refractive index).
    first_order : bool, optional
        If True, uses the first-order contribution for weak-contrast regime
        If False, computes the full Mie coefficient

    Returns
    -------
    complex 
        The computed Mie scattering coefficient 'a_n' for the given parameters.

    Notes
    -----
    - Requires the Bessel functions J (of the first kind), H (Hankel function), and their derivatives dJ, dH.
    - For first_order=True, uses an analytical approximation for weak contrasts
    - For first_order=False, uses the general Mie theory expression for cylinders.
    """

    y = m*x
    if first_order:
        val = J(n-1,x)**2 - 2*(n-1)/x * J(n,x)*J(n-1,x) + (1-2*n**2/x**2)*J(n,x)**2
        return 0.5j * (m**2-1)/(m**2+1)*np.pi*x**2 * val
    else:
        numerator =   dJ(n,y)*J(n,x) - m*J(n,y)*dJ(n,x)
        denominator = dJ(n,y)*H(n,x) - m*J(n,y)*dH(n,x)
        return numerator/denominator

def b_coeff(n, x, m, first_order = False):
    y = m*x

    if first_order:
        val = (J(n-1,x)**2 + 2*(1-2*n**2/x**2)*J(n,x)**2 + J(n+1,x)**2)/4
        return (m**2-1)*0.5j *np.pi*x**2 *val
    else:
        numerator =   m*dJ(n,y)*J(n,x) - J(n,y)*dJ(n,x)
        denominator = m*dJ(n,y)*H(n,x) - J(n,y)*dH(n,x)
        return numerator/denominator

def dsigma_ (ka, eps, thetas,M=50, first_order=False):
    """Dimensionless differential scattering cross section of a single cylinder via Mie Theory.

    Calculates the normalized differential scattering cross section per unit cylinder radius for TE and TM polarizations.

    ka : float
        Dimensionless wavenumber (k * a), where k is the wavenumber in the reference phase and a is the cylinder radius.
    eps : float
        Relative dielectric constant of the cylinder with respect to the background.
    thetas : array-like
        Scattering angles in radians.
    M : int, optional
        Number of Mie coefficients to use in the series expansion (default is 50).
    first_order : bool, optional
        If True, use first-order approximations in dielectric contrast for the Mie coefficients (default is False).

    Returns
    -------
    result : ndarray, shape (len(thetas), 2)
        Differential scattering cross section for each angle:
            result[:, 0] : TE polarization
            result[:, 1] : TM polarization

    Notes
    -----
    The result is normalized by the cylinder radius and wavenumber, i.e., (d\sigma/dθ) / a."""
    m = np.sqrt(eps)
    bn = np.array([ b_coeff(n,ka,m,first_order) for n in range(0,M)])
    an = np.array([ a_coeff(n,ka,m,first_order) for n in range(0,M)])
    # print(bn)
    # print(an)
    result = np.zeros((len(thetas),2))
    for i, theta in enumerate(thetas):
        series_a = 2*np.sum(np.array([an[n]*np.cos(n*theta) for n in range(1,M)]))
        series_a += an[0]

        series_b = 2*np.sum(np.array([bn[n]*np.cos(n*theta) for n in range(1,M)]))
        series_b += bn[0]
        result[i,0] = np.abs(series_a)**2
        result[i,1] = np.abs(series_b)**2
    result *= 2/(np.pi*ka)
    return result

def sigma_ (ka, eps, M=50,first_order=False):
    """Dimensionless total scattering cross section of a single cylinder via Mie Theory.

    Calculates the normalized total scattering cross section per unit cylinder radius for TE and TM polarizations.

    ka : float
        Dimensionless wavenumber (k * a), where k is the wavenumber in the reference phase and a is the cylinder radius.
    eps : float
        Relative dielectric constant of the cylinder with respect to the background.
    M : int, optional
        Number of Mie coefficients to use in the series expansion (default is 50).
    first_order : bool, optional
        If True, use first-order approximations in dielectric contrast for the Mie coefficients (default is False).

    Returns
    -------
    result : ndarray, shape (len(thetas), 2)
        Total scattering cross section for each angle:
            result[:, 0] : TE polarization
            result[:, 1] : TM polarization

    Notes
    -----
    The result is normalized by the cylinder radius and wavenumber, i.e., sigma / a."""

    m = np.sqrt(eps)
    bn = np.array([ b_coeff(n,ka,m,first_order) for n in range(0,M)])
    an = np.array([ a_coeff(n,ka,m,first_order) for n in range(0,M)])
    # print(bn)
    # print(an)
    result = np.array([0.0,0.0])

    # TE:
    result[0] = np.abs(an[0])**2
    result[0] += 2.0*np.sum(np.abs(an[1:])**2)

    # TM: 
    result[1] = np.abs(bn[0])**2
    result[1] += 2.0*np.sum(np.abs(bn[1:])**2)
    result *= 2*np.pi *  2/(np.pi*ka)
    return result
