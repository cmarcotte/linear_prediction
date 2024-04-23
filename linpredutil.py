import h5py
import numpy as np
from docopt import docopt
import matplotlib.pyplot as plt
from scipy.fftpack import diff as fdiff
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq, shift
from scipy.signal import correlate
from scipy.interpolate import InterpolatedUnivariateSpline, PchipInterpolator, splrep, PPoly

def interpolant(x,y,k=3,s=0.0):
	return InterpolatedUnivariateSpline(x,y,k=k,ext=3)

def interper(x,y,z,axis=0):
	M = 2 #np.shape(y)[1]
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
	k = fftfreq(N, d=L/(2*np.pi*N))		# k = 2*pi/L * [0, +1, ..., N/2-1, -N/2, 1-N/2,..., -1]
	k[N//2] = 0.0				# k = 2*pi/L * [0, +1, ..., N/2-1,	0, 1-N/2,..., -1]
	return k

def fftcrosscorr(p, q, L, N):
	'''
	Computes
		r(s) = < p(x-s) | q(x) > = sum_i < p_i(x-s) | q_i(x) >,
	using the cross-correlation function in the scipy stack.
	'''
	
	P = fft(p, axis=0)
	Q = fft(q, axis=0)
	# wikipedia: F(h) = conj(F(f))*F(g)
	res = np.real(ifft(np.conj(P)*Q, axis=0))

	return varsum(res)*(L/N)

def translate(p, s, L, N):
	'''
	Computes the action of the translation operator on the array p of shift s,
		translate(p, s) = T(s)*p(x) = p(x-s)
	Which assumes a periodic array p of length n, where the grid array
		x = [x_0, x_1, ..., x_{N-1}], such that x[N-1]+(x[1]-x[0]) == L
	and uses the Fourier transform to shift by s exactly.
	'''
	
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
			
	return out

def eigenDifferential(w,x):
	# form dw from w
	dw = []
	for m in range(np.shape(w)[1]):
		spl = interpolant(x, w[:,m], k=5)
		spl = spl.derivative(1)
		dw.append(spl(x))
	dw = np.array(dw)
	dw=dw.T # for some reason the array becomes (2,N) rather than (N,2)
	return dw

def defineMUandPHI(x, u, U, w1, w2, v1, v2, L, N, shifty, spl=True, test=True, savedir=""):
	'''
	forms the lambda functions mu(X) and Phi(s)
	x is array of size [N,1]
	u,U,w1,w2,v1,v2 are arrays of size [N,m], m is number of vars
	L is the physical domain length
	N is the length of the domain grid, len(x) == N
	shifty selects the index of the root functions
	'''
		
	if shifty == 1:
		# define generic vectors for mu and Phi defs
		V1 = w1
		V2 = eigenDifferential(w1,x)

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
		V2 = eigenDifferential(v1,x)

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
	testX[:,1:] = 0.0
	
	# ensure testX is smooth spatially by evolving along dudt - d2udx2 = 0, for t=0..1
	testX = fft(testX, axis=0)
	testX = np.reshape(np.exp(-fourierfreqs(N,L)**2),[N,1])*testX
	testX = np.real(ifft(testX, axis=0))
	
	print(f'Testing l={shifty}')

	if test:
		tests, error = test_mu_phi(x, mu, Phi, testX, L, N, x, savedir)
		ind = np.argsort(tests)
		tests = tests[ind]
		error = error[ind]
		
		# compute a tolerance from the maximum absolute error, |e|_inf
		ind = np.ravel(np.where(np.isfinite(error)))
		tol = np.max(np.abs(error[ind]))
	else:
		tol = N*np.spacing(1.0)
	
	return (mu, Phi, tol)

def test_mu_phi(x, mu, Phi, X, L, N, spl=True, savedir=""):
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
	S = np.linspace(L/(M+1),M*L/(M+1),M+1) + 0.0*(L/(M+1))*(2*np.random.rand(M+1)-1.0)
		
	# interpolate this interpolated function Q(s=x) to Q(s=S)
	Q = np.copy(QQ(S))

	# pre-allocate the array values for the rote inner products
	error = np.zeros_like(Q)

	print(f'Testing {len(S)} shift values...')

	# for each s the test set, evaluate the difference
	for n,s in enumerate(S):
		error[n] = innpro(Phi(s), X, L, N)-Q[n]
	
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

	print(f'L∞ error = {np.max(np.abs(error))}')
	
	error = error / (np.max(Q)-np.min(Q))
	
	return (S, error)
	
# evaluate each field on the equispaced grid, form arrays, drop coordinate column
def interpontox(f,x):
	# original field f is expected to be
	#	[z,y,...]
	# we are trying to express y on x, x = [0,1,2,...,N]*L/(N+1).
	# the form of z is a non-uniform discretization of some segment
	# of R, of length l which is unrelated to L.
	# we can not say, a priori, where z is.
	# we must first detect where z is, and shift it to appropriate
	# position to match x.
	
	z = np.copy(f[:,0])
	print("Original domain: [{0},{1}]".format(z[0],z[-1]))
	
	# two usual cases:
	#	f[:,1] is centered at z=0, z = [-a*l, (1-a)*l]
	#	f[:,1] is centered elsewhere, z = [0, l] 
					
	# center field coordinates at z0, 
	z = z-z[0]
	# this transforms case 1 into case 2.
	print("Shifted domain: [{0},{1}]".format(z[0],z[-1]))
	
	# now interpolate f onto x-z[0]
	f = interper(z, f[:,1:], x, axis=0)
	
	# and set everything outside z limits to boundary values
	#ind = np.where(x > z[-1])
	#f[ind,:] = f[ind[0]-1,:]
	#print("x > z[-1]: {0}".format(ind))
	
	#ind = np.where(x < z[0])
	#f[ind,:] = f[ind[-1]-1,:]
	#print("x < z[0]: {0}".format(ind))

	return f

def orthonormalizeEigenfunctions(w1,w2,v1,v2,L,N,x,savedir=""):
	# check bi-orthogonality of eigenfunctions
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

	print("Ortho-Normalizing...")
	v1 = v1/np.sqrt(B[0,0]); av1 = np.sign(v1[np.argmax(np.abs(v1[:,0])),0]); v1 = av1*v1
	v2 = v2/np.sqrt(B[1,1]);
	w1 = w1*np.sqrt(B[0,0])*av1/A[0,0]
	w2 = w2*np.sqrt(B[1,1])/A[1,1]
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
	
	# plot the eigenfunctions
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
	
	
	return (w1,w2,v1,v2)

def rootFilters(SS, QQ, RR, PP, UU, tol):
	# now we construct filters to get a smaller set of roots
	print('Filtering:')
	
	# filter SS such that |QQ(s = SS) | < tol, i.e., mu_l(s) = 0
	ind = np.ravel(np.where(np.abs(QQ(SS)) < tol))
	print(f'\tNon-roots...\n\t\tlen(S) = {len(SS)} -> {len(ind)}')
	SS = SS[ind]
	
	# filter SS such that |RR(s = SS) | > tol, i.e., < w_1(x-s) | X(x)> =/= 0
	ind = np.ravel(np.where(np.abs(RR(SS)) > tol))
	print(f'\tNon-trivial projections...\n\t\tlen(S) = {len(SS)} -> {len(ind)}')
	SS = SS[ind]
	
	# filter SS such that |P(s = SS)| < 1/tol, i.e., P(s) is not ~inf
	ind = np.ravel(np.where(np.abs(PP(SS)) < 1.0/tol))
	print(f'\tFinite perturbation amplitude...\n\t\tlen(S) = {len(SS)} -> {len(ind)}')
	SS = SS[ind]

	# filter SS such that P(s = SS) < -tol
	#ind = np.ravel(np.where(PP(SS) < -tol))
	#print(f'\tNegative perturbation amplitude...\n\t\tlen(S) = {len(SS)} -> {len(ind)}')
	#SS = SS[ind]

	# filter SS such that P(s = SS)*PP''(s = SS) > 0, i.e., local min (P>0) or max (P<0)
	#ind = np.ravel(np.where(PP(SS)*PP.derivative(2)(SS) > 0.0))
	#print(f'\tExtremizing roots...\n\t\tlen(S) = {len(SS)} -> {len(ind)}')
	#SS = SS[ind]
	
	# filter SS such that |P'(s = SS)| < tol, i.e., P(s) is not locally sensitive to s
	#ind = np.ravel(np.where(np.abs(RR.derivative(1)(SS)*PP.derivative(1)(SS)) < tol))
	#print(f'\tExtremizing (P*R)(s)...\n\t\tlen(S) = {len(SS)} -> {len(ind)}')
	#SS = SS[ind]
	
	# filter SS such that |R(s = SS)P'(s = SS) + R'(s = SS)P(s = SS)| < tol, i.e., P(s) is not locally sensitive to s
	#ind = np.ravel(np.where(np.abs(RR(SS)*PP.derivative(1)(SS) + RR.derivative(1)(SS)*PP(SS)) > tol))
	#print(f'\tLocal insensitivity...\n\t\tlen(S) = {len(SS)} -> {len(ind)}')
	#SS = SS[ind]
	
	for S in SS:
		print("S={0:.5E}".format(S))
		for (nf, ff) in zip(['QQ','PP','RR','UU'], [QQ, PP, RR, UU]):
			print(f"\t{nf}(S)={ff(S):+.5E},\t{nf}'(S)={ff.derivative(1)(S):+.5E},\t{nf}''(S)={ff.derivative(2)(S):+.5E}")
	
	return SS
