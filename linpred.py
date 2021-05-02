'''
Usage:
linpred.py [options]

Options:
--N=<N>		size of equispaced grid		[default: 65536]
--L=<L>		physical size of grid
--savedir=<savedir>	saving directory
--ufile=<ufile>	slow wave file
--v1file=<ufile>	v1 file
--v2file=<ufile>	v2 file
--w1file=<ufile>	w1 file
--w2file=<ufile>	w2 file
--Ufile=<Ufile>	initial condition file
--u0file=<u0file>	rest state file
--theta=<theta>	origin of perturbation			[default: 6]
'''

import h5py
import numpy as np
from docopt import docopt
import matplotlib.pyplot as plt
from scipy.fftpack import diff as fdiff
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq, shift
from scipy.signal import correlate
from scipy.interpolate import UnivariateSpline

def interpolant(x,y,k=3,s=0.0):
	return UnivariateSpline(x,y,k=k,s=s)

def interper(x,y,z,axis=0):
	M = np.shape(y)[1]
	out = np.zeros((len(z),M))
	for m in range(M):
		ff = interpolant(x,y[:,m],k=5)
		out[:,m] = ff(z)
	return out

'''
Throughout, N designates the length of the domain grid, e.g.,
	x = np.linspace(0.0,L,N+1)[:N]
so that len(x) == N and x[N-1] + (x[1]-x[0]) == L.
That is, if it existed, x[N+1] == L.
'''

def varsum(p):
	'''
	Compute the sum of p over the variable index,
	Inputs:
		p : [n, m] size array, where n is the domain grid size
					and m is the number of variables.
	Returns the sum over m: out[n] = sum_m p[n, m]
	'''

	return np.sum(p, axis=-1)

def innpro(p, q, L, N):
	'''
	Compute the inner product of p & q using the grid weights,
		< p(x) | q(x) > = sum_{i=0}^{N} w(x_i) * sum_{j} p_j(x_i) * q_j(x_i),
	For an equidistant grid, x = [x_0,...,x_{N-1}], of len(x) == N
		w(x_i) = L/N, 0 <= i <= N-1

	Inputs:
		p : [n, m] size array, where n is the domain grid size and m is the number of variables in the state array.
		q : [n, m] size array, where n is the domain grid size and m is the number of variables in the state array.
		L : physical size of the domain
		N : length of grid array x

	Outputs:
		res is the scalar value of the inner product
	'''

	# first compute the variable index array sum
	#	res[i](x) = sum_i p[i](x) * q[i](x)
	res = varsum(p*q)

	# compute the integral
	#	innpro = sum_j res[j] * (L/N)
	return np.sum(res)*(L/N)

def fourierfreqs(N,L):
	k = fftfreq(N, d=L/(2*np.pi*N))	# k = 2*pi/L * [0, +1, ..., N/2-1, -N/2, 1-N/2,..., -1]
	k[N//2] = 0.0				# k = 2*pi/L * [0, +1, ..., N/2-1,    0, 1-N/2,..., -1]
	return k

def fftcrosscorr(p, q, L, N, fourier=True):
	'''
	Computes
		r(s) = < p(x-s) | q(x) > = sum_i < p_i(x-s) | q_i(x) >,
	using the cross-correlation function in the scipy stack.
	'''

	if fourier:
		P = fft(p, axis=0)
		Q = fft(q, axis=0)
		# wikipedia: F(h) = conj(F(f))*F(g)
		res = np.real(ifft(np.conj(P)*Q, axis=0))
	else:
		# initialize the result of the cross-correlation
		res = np.zeros_like(p)
		# look, scipy.signal.correlate is defined as
		#	z(s) = (x \star y)(s) = \int x(l) \conj{y}(l-s) dl
		# while wiki
		#	(f \star g)(s) 	= \int \conj{f}(t) g(t+s) dt
		#		       	= \int \conj{f}(t-s) g(t) dt
		#		       	= ifft(conj(fft(f)) * fft(g))
		# so then the scipy call should be
		#	correlate(y, x)
		# and when using the mode='same' keyword, we then need to shift
		# the output to account for how scipy orders the indices
		for m in range(np.shape(p)[1]):
			#res[:,m] = shift(correlate(q[:,m], p[:,m], mode='same'),L/2,period=L)
			res[:,m] = np.roll(correlate(q[:,m], p[:,m], mode='same'),N//2)

	return varsum(res)*(L/N)

def translate(p, s, L, N, fourier=True):
	'''
	Computes the action of the translation operator on the array p of shift s,
		translate(p, s) = T(s)*p(x) = p(x-s)
	Which assumes a periodic array p of length n, where the grid array
		x = [x_0, x_1, ..., x_{N-1}], such that x[N-1]+(x[1]-x[0]) == L
	and uses the Fourier transform to shift by s exactly.
	'''
		
	if fourier:
		k = fourierfreqs(N,L)
		T = np.exp(-1j*s*k)			# exp(-1i*s*k[n]) ~ fft(D)[n]
		T = np.reshape(T, [N,1])		# resize for broadcasting multiplication

		# transform along spatial axis
		P = fft(p, axis=0)

		# shift the modes
		P = T*P

		# inverse transform along modal axis
		out = ifft(P, axis=0)

		# make it real
		out = np.real(out)
	else:
		# make a copy of the input array
		out = np.copy(p)
		# ideally, would use scipy.fftpack.shift:
		for m in range(np.shape(p)[1]):
			out[:,m] = shift(out[:,m], -s, period=L)
		
	return out

def defineMUandPHI(x, u, U, w1, w2, v1, v2, L, N, shifty, spl=True, test=True):
	'''
	forms the lambda functions mu(X) and Phi(s)
	x is array of size [N,1]
	u,U,w1,w2,v1,v2 are arrays of size [N,m], m is number of vars
	L is the physical domain length
	N is the length of the domain grid, len(x) == N
	shifty selects the index of the root functions
	'''
	# form dw1 from w1
	dw1 = []
	
	for m in range(np.shape(w1)[1]):
		if spl:
			spl = interpolant(x, w1[:,m], k=5)
			spl = spl.derivative(1)
			dw1.append(spl(x))
		else:
			dw1.append(fdiff(w1[:,m],order=1,period=L))
	dw1 = np.array(dw1)
	dw1=dw1.T # for some reason the array becomes (2,N) rather than (N,2)
	
	# form dv1 from v1
	dv1 = []
	
	for m in range(np.shape(v1)[1]):
		if spl:
			spl = interpolant(x, v1[:,m], k=5)
			spl = spl.derivative(1)
			dv1.append(spl(x))
		else:
			dv1.append(fdiff(v1[:,m],order=1,period=L))
	dv1 = np.array(dv1)
	dv1=dv1.T # for some reason the array becomes (2,N) rather than (N,2)
	
	if shifty == 1:
		# define generic vectors for mu and Phi defs
		V1 = w1
		V2 = dw1

	elif shifty == 2:
		# define generic vectors for mu and Phi defs
		V1 = w1
		V2 = v2

	elif shifty == 3:
		# define generic vectors for mu and Phi defs
		V1 = w1
		V2 = w2

	elif shifty == 4:
		# define generic vectors for mu and Phi defs
		V1 = v1
		V2 = dv1

	elif shifty == 5:
		# define generic vectors for mu and Phi defs
		V1 = v1
		V2 = v2

	elif shifty == 6:
		# define generic vectors for mu and Phi defs
		V1 = v1
		V2 = w2

	# define mu function
	def mu(X):
		f = fftcrosscorr(innpro(V1,u,L,N)*V2-innpro(V2,u,L,N)*V1,X,L,N)
		f+=-fftcrosscorr(V1,U,L,N)*fftcrosscorr(V2,X,L,N)
		f+= fftcrosscorr(V2,U,L,N)*fftcrosscorr(V1,X,L,N)
		return f

	# define Phi function
	def Phi(s):
		f = innpro(translate(V1,s,L,N),translate(u,s,L,N)-U,L,N)*translate(V2,s,L,N)
		f+=-innpro(translate(V2,s,L,N),translate(u,s,L,N)-U,L,N)*translate(V1,s,L,N)
		return f
	
	# before beginning, check that the components are matching and accurate
	testX = 2.0*np.random.rand(np.shape(w1)[0],np.shape(w1)[1])-1.0
	
	# ensure testX is smooth spatially by evolving along dudt - d2udx2 = 0, for t=0..1
	testX = fft(testX, axis=0)
	testX = np.reshape(np.exp(-fourierfreqs(N,L)**2),[N,1])*testX
	testX = np.real(ifft(testX, axis=0))
	
	print(f'Testing shifty={shifty}')

	if test:
		tests, error = test_mu_phi(x, mu, Phi, testX, L, N)
		ind = np.argsort(tests)
		tests = tests[ind]
		error = error[ind]
		
		# compute a tolerance from the error by interpolating the (shifts, error**2)
		# potentially with a smoothed approximation, and compute integral over [0,L]
		tol = np.sqrt(interpolant(tests, error**2, k=5, s=0).integral(0,L))
		print(f'Tolerance based on interpolating L2 error is {tol}.')
		if np.isnan(tol) or np.isinf(tol):
			tol = np.sqrt(interpolant(tests, error**2, k=5, s=len(error)).integral(0,L))
			print(f'Tolerance based on approximate L2 error is {tol}.')
	else:
		tol = np.sqrt(N)*np.spacing(1.0)
	
	return (mu, Phi, tol)

def test_mu_phi(x, mu, Phi, X, L, N, spl=True):
	'''
	This tests the accuracy of the constructed function
		Q(s) = mu(X),
	compared to the rote inner product,
		< Phi(x-s) | X(x) > = < T(s)*Phi(x) | X(x) >,
	across testing values of s spanning [0,L], and returns the error.
	'''
	# first compute Q(s) on the grid s=x,
	Q = mu(X)

	# generate interpolant for this function Q(s=x)
	QQ = interpolant(x, Q)
	
	'''
	# compute the roots of QQ
	if spl:
		S = QQ.roots()
	else:
		ind = np.where(np.sign(Q[:-1]) != np.sign(Q[1:]))[0]
		# this gives the indices before the sign change, so that
		# the actual roots are between [Q[ind[i]], Q[ind[i]+1]]
		# then take y = m*x + b,
		#	b = Q[ind]
		#	m = (Q[ind+1]-Q[ind])/(x[ind+1]-x[ind]),
		#	x = -b/m, for Q[ind] <= x <= Q[ind+1]
		# so that the linear solution is:
		S = x[ind] - Q[ind] * (x[ind+1]-x[ind]) / (Q[ind+1]-Q[ind])

	# perturb the roots of QQ by noise in [-1,+1]*10.0**-10
	S = S + 1e-14*(2.0*np.random.rand(len(S))-1.0)

	# generate more testing values of s
	S = np.insert(S, 0, L*np.random.rand(2048))

	# make sure S is size [M,1]
	S = np.ravel(np.sort(S[:]))
	
	# while S is large, trim it
	while len(S) > 1024:
		S = S[::2]
	'''
	
	# how many samples
	M=256
	
	# sample shifts
	S = np.linspace(L/(M+1),M*L/(M+1),M+1) + (L/(M+1))*(2*np.random.rand(M+1)-1.0)
		
	# interpolate this interpolated function Q(s=x) to Q(s=S)
	Q = np.copy(QQ(S))

	# pre-allocate the array values for the rote inner products
	error = np.zeros_like(Q)

	print(f'Testing {len(S)} shift values...')

	# for each s the test set, evaluate the difference
	for n,s in enumerate(S):

		if np.mod(n,int(np.ceil(np.log2(len(S))))) == 0:
			print(f'\t{n:04d} of {len(S):04d}...')

		error[n] = innpro(Phi(s), X, L, N)-Q[n]

	print(f'L∞ error = {np.max(np.abs(error))}')

	fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6,10))
	fig.suptitle('Testing shifts')
	ax[0].plot(x, Phi(0.0), '-')
	ax[0].legend(loc=0)
	ax[1].plot(x, QQ(x), '-k', label=r'$\mu(s)$')
	ax[1].plot(S, error+Q, '.r', label=r'$\langle \Phi(x-s) | X(x) \rangle$')
	ax[1].legend(loc=0)
	ax[2].plot(S, np.abs(error), '.r')
	ax[2].set_yscale('log')
	ax[2].set_xlabel(r'$s$')
	ax[2].set_ylabel(r'$|\mu - \langle \Phi(x-s) | X(x) \rangle|(s)$')
	plt.savefig(f'{savedir}/shift_error.svg',bbox_inches='tight')
	plt.close()
	
	error = error / (np.max(Q)-np.min(Q))
	
	return (S, error)

def threshold_linpred(x, u, U, w1, mu, Phi, tol, X, L, N, spl=True):
	'''
	Two most relevant variables are:
		Q(s) 	= mu(X)
			= < Phi(x-s) | X(x) >
			= < T(s) Phi(x) | X(x) >
		P(s)  	= < w_1(x-s) | u(x-s) - U(x) > / < w_1(x-s) | X(x) >
			= < T(s)*w_1(x) | T(s)*u(x) - U(x) > / < T(s)*w_1(x) | X(x) >
			= (< w_1(x) | u(x) > - < T(s)*w_1(x) | U(x) > ) / < T(s)*w_1(x) | X(x) >
	note that X(x) and U(x) are fixed on the grid.
	'''

	# first form Q(s) = mu(X)
	Q = mu(X)

	# next form R(s) = < w_1(x-s) | X(x) >
	R = fftcrosscorr(w1, X, L, N)

	# next form P(s) = < w_1(x-s) | u(x-s) - U(x) > / < w_1(x-s) | X(x) >
	P = ( innpro(w1, u, L, N) - fftcrosscorr(w1, U, L, N) ) / R

	# form interpolants of Q, P, and R only on values of x which are reliable
	# for Q & R, it is expected that ind = [0:N], but some R == 0, so some P is inf
	ind  = np.ravel(np.where(np.isfinite(Q)))
	QQ = interpolant(x[ind], Q[ind], k=3) # k=3 because need root-finding
	ind  = np.ravel(np.where(np.isfinite(R)))
	RR = interpolant(x[ind], R[ind], k=5)
	ind  = np.ravel(np.where(np.isfinite(P)))
	PP = interpolant(x[ind], P[ind], k=5)

	# find roots of QQ
	if spl:
		SS = QQ.roots()
	else:
		# alternatively using the grid values alone:
		SS = np.where(np.sign(Q[:-1]) != np.sign(Q[1:]))[0]
		# this gives the indices before the sign change, so that
		# the actual roots are between [Q[ind[i]], Q[ind[i]+1]]
		# then take y = m*x + b,
		#	b = Q[ind]
		#	m = (Q[ind+1]-Q[ind])/(x[ind+1]-x[ind]),
		#	x = -b/m, for Q[ind] <= x <= Q[ind+1]
		# so that the linear solution is:
		SS = x[SS] - Q[SS] * (x[SS+1]-x[SS]) / (Q[SS+1]-Q[SS])

	# now we construct filters to get a smaller set of roots
	print('Filtering:')

	# filter SS such that P(s = SS)*PP''(s = SS) > 0, i.e., local min (P>0) or max (P<0)
	#ind = np.ravel(np.where(PP(SS)*PP.derivative(2)(SS) > 0.0))
	#print(f'\tlen(S) = {len(SS)} -> {len(ind)}')
	#SS = SS[ind]

	# filter SS such that |QQ'(s = SS)| > tol, i.e., an "isolated" zero
	ind = np.ravel(np.where(np.abs(QQ.derivative(1)(SS)) > tol))
	print(f'Isolating roots...\n\tlen(S) = {len(SS)} -> {len(ind)}')
	SS = SS[ind]

	# filter SS such that |RR(s = SS) | > tol, i.e., < w_1(x-s) | X(x)> =/= 0
	ind = np.ravel(np.where(np.abs(RR(SS)) > tol))
	print(f'Non-trivial projections...\n\tlen(S) = {len(SS)} -> {len(ind)}')
	SS = SS[ind]

	# filter SS such that |P(s = SS)| < 1/tol, i.e., P(s) is not ~inf
	ind = np.ravel(np.where(np.abs(PP(SS)) < 1.0/tol))
	print(f'Finite perturbation amplitude...\n\tlen(S) = {len(SS)} -> {len(ind)}')
	SS = SS[ind]

	# order SS by decreasing |R(SS)|
	ind = np.argsort(RR(SS))[::-1]
	SS = SS[ind]

	# compute the values of the perturbation based on the interpolant
	US = PP(SS)

	return (SS, US, Q, R, P)

def generatelinpredcurve(x, u, U, w1, mu, Phi, tol, XX, L, N):
	'''
	x is array of size [N,1]
	u,U,w1,w2,v1,v2 are arrays of size [N,m], m is number of vars
	L is the physical domain length
	N is the length of the domain grid
	shifty selects the index of the root functions
	'''

	# form the perturbation widths
	XS = np.logspace(np.log10(2*L/N), np.log10(L/2), 256)

	# create lists for the all the shifts and amplitudes
	SS = []
	US = []

	# compute the shift and amplitude for each perturbation width
	for n,xs in enumerate(XS):

		X = XX(xs)
		s,us,Q,R,P = threshold_linpred(x, u, U, w1, mu, Phi, tol, X, L, N)

		SS.append(s)
		US.append(us)

		if np.mod(n,int(np.ceil(np.log2(len(XS))))) == 0:

			# form interpolants of Q, P, and R
			ind  = np.ravel(np.where(np.isfinite(Q)))
			QQ = interpolant(x[ind], Q[ind])
			ind  = np.ravel(np.where(np.isfinite(R)))
			RR = interpolant(x[ind], R[ind])
			ind  = np.ravel(np.where(np.isfinite(P)))
			PP = interpolant(x[ind], P[ind])

			fig, ax = plt.subplots(3, 1, sharex=True, figsize=(3,8))
			fig.suptitle('Linear theory prediction\n'+r'$x_s={0:2.2f}$'.format(xs))
			Qnrm = np.max(np.abs(Q))
			ax[0].plot(x, Q/Qnrm, '-k', label=r'$Q(s)$')

			ax[0].set_yscale('symlog')
			ax[0].set_ylim([-1.1,+1.1])
			ax[0].set_yticks([0.0])
			ax[0].set_ylabel(r"$Q(s)/\sup_s \, Q(s)$")

			ax[1].plot(x, R, '-k', label=r'$R(s)$')
			ax[1].set_ylabel(r"$R(s)$")

			ax[2].plot(x, P, '-k', label=r'$P(s)$')
			ax[2].set_yscale('symlog')
			ax[2].set_ylabel(r"$P(s)$")

			plt.xlabel(r'$s$')

			try:
				ax[0].plot(s, QQ(s)/Qnrm, '.k')
				ax[1].plot(s, RR(s), '.k')
				ax[2].plot(s, PP(s), '.k')

				ax[0].plot(s[0], QQ(s[0])/Qnrm, 'or', markerfacecolor='none')
				ax[1].plot(s[0], RR(s[0]), 'or', markerfacecolor='none')
				ax[2].plot(s[0], PP(s[0]), 'or', markerfacecolor='none')
				
				ind = np.abs(R) > tol
				plim = np.max(np.abs(P[ind]))
				if np.isnan(plim) or np.isinf(plim):
					plim = 0.5/tol
				ax[2].set_ylim([-2*plim,+2*plim])
			except:
				pass

			plt.tight_layout()
			plt.savefig(f'{savedir}/linear_theory.svg',bbox_inches='tight')
			plt.close()

	return (XS, SS, US)

def computecurves(x, U, u, v1, v2, w1, w2, XX, L, N, theta, shifty, savedir='./'):
	'''

	Takes as input:
		the equispaced grid, x
		the interpolated initial state, U
		the interpolated reference wave, u
		the interpolated reference eigenfunction, v1
		the interpolated reference eigenfunction, v2
		the interpolated reference eigenfunction, w1
		the interpolated reference eigenfunction, w2
		the definition of the perturbation function, XX(xs) = X(x;xs)
		the physical length of the domain of x, L
		the numerical length of the equispaced grid x, N
		the save directory, savedir

	And the function saves the files crit_{n}.h5 to savedir, for n=1,...,6

	and the function returns the curve data (XS, SS, US)
	'''

	# generate a figure showing the positioning of the state, perturbation, reference wave
	fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(3,8))

	ax[0].plot(x, U, '-')
	ax[0].set_title(r'$u(t=0,x)$')

	xs=np.random.rand()*(L/2.0)
	ax[1].plot(x, XX(xs, theta=theta), '-')
	ax[1].set_title(r'$X(x;x_s={0:2.2f}L)$'.format(xs/L))

	ax[2].plot(x, u, '-')
	ax[2].set_title(r'$\hat{u}(x)$')

	ax[2].set_xlabel(r'$x$')

	plt.savefig(f'{savedir}/configures.svg',bbox_inches='tight')
	plt.close()

	# get mu and Phi funcs
	mu, Phi, tol = defineMUandPHI(x, u, U, w1, w2, v1, v2, L, N, shifty)

	# compute the critical curve (XS, SS, US)
	XS, tmpSS, tmpUS = generatelinpredcurve(x, u, U, w1, mu, Phi, tol, XX, L, N)

	xl = [len(tmpSS[n]) for n in range(len(tmpSS))]

	SS = np.ones((len(XS), max(xl)))*np.nan
	US = np.ones((len(XS), max(xl)))*np.nan

	for n in range(len(tmpSS)):
		if xl[n] > 0:
			SS[n,:xl[n]] = np.copy(tmpSS[n])
			US[n,:xl[n]] = np.copy(tmpUS[n])

	# plot the predicted critical curve
	fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6,6))
	if len(SS) > 0:
		ax[0].plot(XS, US, '.')
		ax[0].set_yscale('symlog')
		ax[0].set_ylabel(r'$U_s$')
		ax[1].plot(XS, SS, '.')
		ax[1].set_xscale('log')
		ax[1].set_ylim([0,L])
		ax[1].legend(loc=0)
		ax[1].set_xlabel(r'$x_s$')
		ax[1].set_ylabel(r'$s$')
		plt.savefig(f'{savedir}/crit_{shifty}.svg')
	plt.close()

	# save results to file
	with h5py.File(f'{savedir}/crit_{shifty}.h5', mode='w') as savefile:

		savefile.create_dataset('x', 	data=x)

		savefile.create_dataset('U', 	data=U)

		savefile.create_dataset('u', 	data=u)
		savefile.create_dataset('v1', 	data=v1)
		savefile.create_dataset('v2', 	data=v2)
		savefile.create_dataset('w1', 	data=w1)
		savefile.create_dataset('w2', 	data=w2)

		savefile.create_dataset('Phi',	data=Phi(0.0))

		savefile.create_dataset('XS', 	data=XS)
		savefile.create_dataset('SS', 	data=SS)
		savefile.create_dataset('US', 	data=US)
		
		savefile.create_dataset('TH',		data=theta)
	
	return (XS, SS, US)

# now for the main part
if __name__ == '__main__':
		
	args = docopt(__doc__)

	N = int(args['--N'])
	L = float(args['--L'])

	savedir = args['--savedir']

	u0file = args['--u0file']

	Ufile = args['--Ufile']
	
	ufile = args['--ufile']
	
	w1file = args['--w1file']
	w2file = args['--w2file']
	v1file = args['--v1file']
	v2file = args['--v2file']
	
	theta = int(args['--theta'])

	# get u0 from specific file
	u0 = np.loadtxt(u0file)

	# get u, w1, w2, v1, v2, from files
	u = np.loadtxt(ufile)

	w1 = np.loadtxt(w1file)
	w2 = np.loadtxt(w2file)
	v1 = np.loadtxt(v1file)
	v2 = np.loadtxt(v2file)

	# get U from specific file -- may correspond to:
	# fast pulse, initial condition, rest state, etc.
	U = np.loadtxt(Ufile)

	# form x, the equispaced grid of length N on which the fields defined
	x = np.linspace(0.0,L,N+1)[:N]

	# evaluate each field on the equispaced grid, form arrays, drop coordinate column
	def interpontox(f):
		# original field is expected to be
		#	[z,y,...]
		# we are trying to express y on x, x = [0,1,2,...,N]*L/(N+1).
		# the form of z is a non-uniform discretization of some segment
		# of R, of length l which is unrelated to L.
		# we can not say, a priori, where z is.
		# we must first detect where z is, and shift it to appropriate
		# position to match x.
		
		z = np.copy(f[:,0])
		
		# two usual cases:
		#	f[:,1] is centered at z=0, z = [-a*l, (1-a)*l]
		#	f[:,1] is centered elsewhere, z = [0, l] 
						
		# center field coordinates at z0, 
		z = z-z[0]
		# this transforms case 1 into case 2.
		
		# now interpolate f onto x-z[0]
		f = interper(z, f[:,1:], x, axis=0)
		
		# and set everything outside z limits to boundary values
		ind = np.where(x > z[-1])
		f[ind,:] = f[ind[-1]-1,:]
		ind = np.where(x < z[0])
		f[ind,:] = f[ind[-1]-1,:]

		return f
	
	u = interpontox(u)
	
	w1 = interpontox(w1)

	w2 = interpontox(w2)

	v1 = interpontox(v1)

	v2 = interpontox(v2)

	U = interpontox(U)
	
	# since u0'(x) == 0, we broadcast on to the domain of x
	u0 = np.ones((N,1))*np.reshape(u0,[1,len(u0[:])])
	
	# check normalization of eigenfunctions
	A = np.zeros((2,2)); B = np.zeros((2,2));
	A[0,0] = innpro(w1,v1,L,N)
	A[0,1] = innpro(w1,v2,L,N)
	A[1,0] = innpro(w2,v1,L,N)
	A[1,1] = innpro(w2,v2,L,N)
	
	B[0,0] = innpro(v1,v1,L,N)
	B[0,1] = innpro(v1,v2,L,N)
	B[1,0] = innpro(v2,v1,L,N)
	B[1,1] = innpro(v2,v2,L,N)
	
	print(f" <w_i | v_j > = \n\t{A[0,:]}\n\t{A[1,:]}")
	print(f" <v_i | v_j > = \n\t{B[0,:]}\n\t{B[1,:]}")
	
	fig, ax = plt.subplots(2, 2, sharex=True, figsize=(6,6))
	ax[0,0].plot(x, w1)
	ax[0,0].set_title(r'$w_1$')
	ax[0,1].plot(x, w2)
	ax[0,1].set_title(r'$w_2$')
	ax[1,0].plot(x, v1)
	ax[1,0].set_title(r'$v_1$')
	ax[1,1].plot(x, v2)
	ax[1,1].set_title(r'$v_2$')
	plt.savefig(f'{savedir}/eigen.svg')
	plt.close()
	
	# define the perturbation origin
	x0 = np.minimum(np.maximum(0.0,L/2 + (theta-6)*L/20),L)
	
	# define the perturbation shape function, XX(xs) -> X(x)
	# this implies, e.g., that theta is fixed at definition
	def XX(xs,theta=x0):
		f = np.zeros_like(U)
		f[:,0]+= (1.0+np.sign(x-theta+xs/2))
		f[:,0]*= (1.0-np.sign(x-theta-xs/2))
		f[:,0]*= 0.25
		#f[:,1]+= f[:,0]*(np.sign(x-theta))
		return f
	
	# plot the predicted critical curve
	fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6,6))

	# compute the critical curves
	for shifty in range(1,7):
		XS, SS, US = computecurves(x, U, u, v1, v2, w1, w2, XX, L, N, x0, shifty, savedir=savedir)
		
		ax[0].plot(XS, US[:,0], '.', label=r'$U_{0}$'.format(shifty), markersize=7-shifty)
		ax[0].set_yscale('symlog')
		ax[0].set_ylabel(r'$U_s$')
		ax[0].legend(loc=0)
		ax[1].plot(XS, SS[:,0], '.', label=r'$s_{0}$'.format(shifty), markersize=7-shifty)
		ax[1].set_xscale('log')
		ax[1].set_ylim([0,L])
		ax[1].legend(loc=0)
		ax[1].set_xlabel(r'$x_s$')
		ax[1].set_ylabel(r'$s$')
		plt.savefig(f'{savedir}/crit_all.svg')
	plt.close()
	
