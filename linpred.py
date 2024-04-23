'''
Usage:
linpredingredients.py [options]

Options:
--N=<N>			size of equispaced grid			[default: 65536]
--L=<L>			physical size of grid
--savedir=<savedir>	saving directory
--ufile=<ufile>		slow wave file
--v1file=<ufile>	v1 file
--v2file=<ufile>	v2 file
--w1file=<ufile>	w1 file
--w2file=<ufile>	w2 file
--Ufile=<Ufile>		initial condition file
--u0file=<u0file>	rest state file
--theta=<theta>		origin of perturbation			[default: 0]
'''

import h5py
import numpy as np
from docopt import docopt
import matplotlib.pyplot as plt
from scipy.fftpack import diff as fdiff
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq, shift
from scipy.signal import correlate
from scipy.interpolate import InterpolatedUnivariateSpline, PchipInterpolator, splrep, PPoly
from linpredutil import *

plt.style.use('seaborn-v0_8-paper')
import matplotlib
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9

# now for the main part
if __name__ == '__main__':
		
	args = docopt(__doc__)
	print(args)
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
	theta = float(args['--theta'])

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

	# interpolate all the inputs onto x
	u = interpontox(u[:,0:-1],x)
	w1 = interpontox(w1,x)
	w2 = interpontox(w2,x)
	v1 = interpontox(v1,x)
	v2 = interpontox(v2,x)
	U = interpontox(U[:,0:-1],x)
		
	# since u0'(x) == 0, we broadcast on to the domain of x
	u0 = np.ones((N,1))*np.reshape(u0,[1,len(u0[:])])

	# orthonormalize the eigenfunctions---not strictly necessary because
	# the linear problem is scale free, but improve conditioning
	w1, w2, v1, v2 = orthonormalizeEigenfunctions(w1, w2, v1, v2, L, N, x, savedir=savedir)
	
	# define the perturbation origin
	# old theta map:
	#thmap = lambda th: (L/2 + (th-6)*L/20)
	thmap = lambda th: x[np.argmax(U[:,0])]+th
	x0 = np.minimum(np.maximum(0.0,thmap(theta)),L)

	# define the perturbation shape function, XX(xs) -> X(x)
	# this implies, e.g., that theta is fixed at definition
	def XX(xs,theta=x0):
		f = np.zeros_like(U)
		f[:,0]+= (1.0+np.sign(x-theta+xs/2))
		f[:,0]*= (1.0-np.sign(x-theta-xs/2))
		f[:,0]*= 0.25
		return f

	# form numerator terms of U(s; x_s, theta) (invariant wrt x_s)
	P1 = innpro(w1, u, L, N); print("P1 = {0:.5E}".format(P1))
	P2 = fftcrosscorr(w1, U, L, N)
	
	# reset s-limits
	slims = [0,L]
	
	# reset U-limits
	ulims = 0
	
	# initial tolerance (very small)
	tol = 1e-10
	
	# set asymptotic s solution
	S0 = [np.nan, np.nan, np.nan]
	
	# range of xs values
	xs_range = np.unique(np.round(np.logspace(np.log10(2*L), 0, 257)))[::-1]
	
	# accumulating list of (xs, s, us) for each shifty
	shifty_range = [1,2,3]
	accum = [np.nan*np.zeros((len(xs_range),3)) for shifty in shifty_range]	
	
	# loop over widths
	for (nxs, xs) in enumerate(xs_range):
				
		# wide pert
		X = XX(xs)

		# form terms in U(s; theta, xs) and write to file?
		R = fftcrosscorr(w1, X, L, N)
		
		# interpolants
		PP = PchipInterpolator(x, P1-P2, extrapolate=False);
		RR = PchipInterpolator(x, R, extrapolate=False);
		inds = np.where(np.isfinite((P1-P2)/R))
		UU = PchipInterpolator(x[inds], (P1-P2[inds])/R[inds], extrapolate=False);
				
		fig, axs = plt.subplots(4, len(shifty_range)+1, figsize=(11,len(shifty_range)*7/3), sharex='col', layout='compressed')
		fig.suptitle(r'$\theta='+f'{theta:2.0f}'+r'$, $x_s = '+f'{xs:2.0f}'+r'$')
		axs[0,0].plot(x, X); 							axs[0,0].set_ylabel(r'$\check{\mathbf{X}}(x-\theta; x_s)$')
		axs[1,0].plot(x, U); 							axs[1,0].set_ylabel(r'$\check{\mathbf{u}}(x)$')
		axs[2,0].plot(x, u); 							axs[2,0].set_ylabel(r'$\hat{\mathbf{u}}(x)$')
		axs[3,0].plot(x, w1);							axs[3,0].set_ylabel(r'$\hat{\mathbf{w}}_1(x)$')
		axs[-1,0].set_xlabel(r"$x$")
		axs[-1,0].set_xlim((x[0],x[-1]))
		
		axs[0,1].set_ylabel(r'$\check{\mu}_l(s; \theta, x_s)$')
		axs[1,1].set_ylabel(r'$\langle \hat{\mathbf{w}}_1(\xi) | \hat{\mathbf{u}}(\xi) - \check{\mathbf{u}}(\xi+s) \rangle$')
		axs[2,1].set_ylabel(r'$\langle \hat{\mathbf{w}}_1(\xi) | \check{\mathbf{X}}(\xi+s-\theta; x_s) \rangle$')
		axs[3,1].set_ylabel(r'$\check{U}(s; \theta, x_s)$'); 	
		
		for shifty in shifty_range:
		
			# get mu and Phi funcs
			mu, Phi, tmp = defineMUandPHI(x, u, U, w1, w2, v1, v2, L, N, shifty, savedir=savedir, test = True if nxs == 0 else False)
			if nxs == 0:
				tol = np.max([tol, tmp])
			
			# form mu_l(s) and write to file
			Q = mu(X)
			Q = Q/np.max(np.abs(Q))
		
			# interpolants for root-finding
			QQ = PchipInterpolator(x, Q, extrapolate=False); 
			SS = QQ.roots()
			
			# prepare axes
			axs[0,shifty].sharex(axs[0,1])
			axs[3,shifty].set_yscale('symlog', linthresh=1e-1)
			for n in range(0,4):
				axs[n,shifty].plot(x, np.zeros_like(x), '--k', linewidth=0.5)
				if shifty > 1:
					axs[n,shifty].sharey(axs[n,1])
					axs[n,shifty].tick_params(labelleft=False)
			axs[0,shifty].set_ylim([-1,1])
			axs[1,shifty].plot(x,P1-P2, '-k')				
			axs[2,shifty].plot(x, innpro(w1, XX(2*L), L, N)*np.ones_like(x), ':k', linewidth=0.5)
			axs[2,shifty].plot(x, R, '-k')
			axs[3,shifty].plot(x, (P1-P2)/R, '-k')	
			
			# filter the roots
			SS = rootFilters(SS, QQ, RR, PP, UU, tol)
			
			axs[0,shifty].plot(x, Q, f'-k', label=r'$\check{\mu}_'+f'{shifty}'+r'(s; \theta, x_s)$'); 	
			axs[0,shifty].plot(SS, QQ(SS), f'.k', markersize=7);
			axs[1,shifty].plot(SS, PP(SS), f'.k', markersize=7);
			axs[2,shifty].plot(SS, RR(SS), f'.k', markersize=7);
			axs[3,shifty].plot(SS, UU(SS), f'.k', markersize=7);
			axs[0,shifty].legend(loc='upper right')
						
			if len(SS) > 0:
				# if asymptotic
				if nxs == 0:
					# find initial s-limits for reasonable roots
					slims[0] = np.max([slims[0], np.min(SS)-250.0])
					slims[1] = np.min([slims[1], np.max(SS)+250.0])
					
					# determine asymptotic solution by sign of U(s) < 0:
					ind = np.ravel(np.where(UU(SS) < 0.0))
					ndi = np.argmin(UU(SS[ind]))
					S0[shifty-1] = SS[ind[ndi]]
					SS = SS[ind[ndi]]
					accum[shifty-1][nxs,0] = xs
					accum[shifty-1][nxs,1] = SS
					accum[shifty-1][nxs,2] = UU(SS)
				elif np.isfinite(S0[shifty-1]):
					ind = np.argmin(np.abs(S0[shifty-1]-SS))
					S1 = SS[ind]
					if np.abs(S1-S0[shifty-1]) < 1.0:
						S0[shifty-1] = S1
						SS = SS[ind]
						accum[shifty-1][nxs,0] = xs
						accum[shifty-1][nxs,1] = SS
						accum[shifty-1][nxs,2] = UU(SS)
					else:
						S0[shifty-1] = np.nan
						SS = []
						accum[shifty-1][nxs,0] = xs
				else:
					S0[shifty-1] = np.nan
					SS = []
					accum[shifty-1][nxs,0] = xs
			axs[0,shifty].plot(SS, QQ(SS), f'.C{shifty-1}', markersize=6);
			axs[1,shifty].plot(SS, PP(SS), f'.C{shifty-1}', markersize=6);
			axs[2,shifty].plot(SS, RR(SS), f'.C{shifty-1}', markersize=6);
			axs[3,shifty].plot(SS, UU(SS), f'.C{shifty-1}', markersize=6);
			axs[-1,shifty].set_xlabel(r"$s$")
			
		ind = np.ravel(np.where( (x >= np.min(slims)) * (x<= np.max(slims)) ))
		tmp = int(np.ceil(np.log10(np.max(np.abs(UU(x[ind]))))))
		ulims = np.min([3,np.max([ulims, tmp])])
		axs[0,1].set_xlim(slims)
		axs[3,1].set_ylim([-10.0**ulims, +10.0**ulims])
		fig.align_ylabels(axs[:,0])
		fig.align_ylabels(axs[:,1])
		plt.savefig(f"{savedir}/{nxs:03d}.svg")
		plt.close()
		
	for shifty in shifty_range:
		np.savetxt(f"{savedir}/{shifty}.dat", accum[shifty-1])
