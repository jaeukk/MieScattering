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

def a_coeff(n, x, m):
	y = m*x
	numerator =   dJ(n,y)*J(n,x) - m*J(n,y)*dJ(n,x)
	denominator = dJ(n,y)*H(n,x) - m*J(n,y)*dH(n,x)
	return numerator/denominator

def b_coeff(n, x, m):
	y = m*x
	numerator =   m*dJ(n,y)*J(n,x) - J(n,y)*dJ(n,x)
	denominator = m*dJ(n,y)*H(n,x) - J(n,y)*dH(n,x)
	return numerator/denominator

def dsigma_ (ka, eps, thetas,M=50):
	"""Dimensionless Differential scattering cross section of a single cylinder via Mie Theory, i.e., 
	$$ \dv{\sigma}{\theta} / a. $$

	Parameters
	----------
	ka
		Dimensionless wavenumber, k = wavenumber in the reference phase, a = cylinder radius
	eps
		Relative dielectric cosntant of the cylinder to the background.
	thetas
		Scattering angles in radian
	M, optional
		The number of coefficients, by default 50

	Result: n x 2 float array
	----------
	Result[:,0] = differential scattering cross section for TE polarization
	Result[:,1] = differential scattering cross section for TM polarization
	"""
	m = np.sqrt(eps)
	bn = np.array([ b_coeff(n,ka,m) for n in range(0,M)])
	an = np.array([ a_coeff(n,ka,m) for n in range(0,M)])
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

def sigma_ (ka, eps, M=50):
	"""Dimensionless scattering cross section of an infinitely long cylinder. Divided by the cylinder radius.

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
		[TE, TM]
	"""

	m = np.sqrt(eps)
	bn = np.array([ b_coeff(n,ka,m) for n in range(0,M)])
	an = np.array([ a_coeff(n,ka,m) for n in range(0,M)])
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
