import scipy.special as sp
import numpy as np

""" Python code to evaluate the Mie coefficients for a single dielectric sphere. """

def spherical_hankel2(n,x):
    return sp.spherical_jn(n,x)-1j*sp.spherical_yn(n,x)

def psi(n,x):
    return x*sp.spherical_jn(n,x)
def zeta(n,x):
    return x*spherical_hankel2(n,x)

def dpsi(n,x):
    """ derivtive of psi(n,x) function"""
    return 0.5*(psi(n-1,x)-psi(n+1,x)+sp.spherical_jn(n,x))
def dzeta(n,x):
    """ derivtive of zeta(n,x) function"""
    return 0.5*(zeta(n-1,x)-zeta(n+1,x)+spherical_hankel2(n,x))

""" orientational """
def arr_pi(theta:float, n_max:int):
    x = np.cos(theta)
    return np.array([sp.lpmv(1, n, x) for n in range(1,n_max)])/np.sin(theta)

def arr_tau(theta:float, n_max:int):
    x = np.cos(theta)
    dPn = lambda n, x : (-(1+n)*x*sp.lpmv(1,n,np.cos(theta)) + n*sp.lpmv(1,1+n,x))/(x**2-1) # derivative of the associated Legendre polynomial
    list = np.array([ dPn(n,x)  for n in range(1,n_max)])
    return -np.sin(theta)*list

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


def dsigma_ (ka, eps, thetas,M=50):
    """Dimensionless Differential scattering cross section of a single sphere via Mie Theory, i.e., 
    $$ \dv{\sigma}{\Omega} / a^2 $$

    Parameters
    ----------
    ka
        Dimensionless wavenumber, k = wavenumber in the reference phase, a = cylinder radius
    eps
        Relative dielectric cosntant of the cylinder to the background.
    thetas
        Array of scattering angles in radian
    M, optional
        The number of coefficients, by default 50

    Result: n x 1 float array
    ----------
    Result = differential scattering cross section for transverse polarization
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
    """Dimensionless scattering cross section of a single sphere. Divided by the square of sphere radius.

    Parameters
    ----------
    ka
        _description_
    eps
        _description_
    M, optional
        _description_, by default 50

    Returns
    -------
        []
    """
    m = np.sqrt(eps)
    bn = np.array([ b_coeff(n,ka,m) for n in range(1,M)])
    an = np.array([ a_coeff(n,ka,m) for n in range(1,M)])
    result = 0.0

    if len(ka) > 1:
        result = np.array([np.inner(np.abs(an[:,i])**2+np.abs(bn[:,i])**2, 2*np.arange(1,M)+1) for i in range(len(ka))])        
    else:
        result = np.inner(np.abs(an)**2+np.abs(bn)**2, 2*np.arange(1,M)+1)
    result *= 2*np.pi/ka**2
    return result
