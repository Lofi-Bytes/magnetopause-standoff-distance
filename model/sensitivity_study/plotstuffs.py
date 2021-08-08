# -*- coding: utf-8 -*-
"""
Created on Thursday March 13 2014
Magnetopause Location
@author: Jillian S. Estrella
"""

#***********************************************************************
# Importing everything python needs in order to be smart.
#***********************************************************************
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Wedge
from matplotlib.path import Path

#***********************************************************************
# Make Plots!
#***********************************************************************
#***********************************************************************
# Define how to make an "Earth"
#***********************************************************************
def dual_half_circle(center, radius, angle=0, ax=None, colors=('w','k'), **kwargs):
	"""
	Add two half circles to the axes *ax* (or the current axes) with the
	specified facecolors *colors* rotated at *angle* (in degrees).
	"""
	if ax is None:
		ax = plt.gca()
	theta1, theta2 = angle, angle + 180
	w1 = Wedge(center, radius, theta1, theta2, fc=colors[1], **kwargs)
	w2 = Wedge(center, radius, theta2, theta1, fc=colors[0], **kwargs)
	for wedge in [w1, w2]:
		ax.add_artist(wedge)
	return [w1, w2]

def plotstuffs(x,y,Bz,Dp):
	labels = []
	for l in range(Bz.size):
		labels.append('$B_{z} = %10.2f [nT]$, $P_{dyn} = %10.2f [nPa]$' % (Bz[l],Dp[l]))
	fig    = plt.figure(figsize=[7.06,7.06])
	ax     = fig.add_subplot(111)
	#ax.set_title('Magnetopause Location \n Sensitivity to $P_{dyn}$')
	#ax.set_title('Magnetopause Location \n Sensitivity to $B_{z}$')
	ax.set_xlabel('$x$ $[R_{E}]$')
	ax.set_ylabel('$R$ $[R_{E}]$')
	ax.set_xlim([15,-20])
	ax.set_ylim([0,-35])
	ax.minorticks_on()
	dual_half_circle((0, 0), radius=1.0, angle=90, ax=ax)
	#ax.plot(GOES_X,GOES_Y, 'o')
	#for i in range(GOES_X.size):
	#	ax.plot(GOES_X/6378.1,GOES_Y/6378.1)
	for j in range(x[0,:].size):
		ax.plot(x[:,j], -y[:,j])
	ax.legend(labels, bbox_to_anchor=(0.02, 0.98), loc=2, borderaxespad=0., fancybox=True, shadow=True)
	#fig.savefig('./figures/dDp.pdf')
	fig.savefig('./figures/dBz.pdf')
	plt.close()

def plotrvsDp(r,Dp):
	labels = ['$B_{z} =  [nPa]$']
	fig    = plt.figure(figsize=[7.06,7.06])
	ax     = fig.add_subplot(111)
	#ax.set_title('Magnetopause Location \n Sensitivity to $P_{dyn}$')
	ax.set_xlabel('$P_{dyn} [nPa]$')
	ax.set_ylabel('$r [R_{E}]$')
	ax.set_xlim([4,19])
	ax.set_ylim([5.6,8])
	ax.minorticks_on()
	ax.axhline(y=6.6, linestyle='--')
	for k in range(r[0,:].size):
		ax.plot(Dp, r[:,k])
	#ax.legend(labels, bbox_to_anchor=(0.69, 0.98), loc=2, borderaxespad=0., fancybox=True, shadow=True)
	fig.savefig('./figures/rvsDp/rvsDp.pdf')
	plt.close()

def plotrvsBz(r,Bz):
	fig    = plt.figure(figsize=[7.06,7.06])
	ax     = fig.add_subplot(111)
	#ax.set_title('Magnetopause Location \n Sensitivity to $B_{z}$')
	ax.set_xlabel('$B_{z} [nT]$')
	ax.set_ylabel('$r [R_{E}]$')
	#ax.set_xlim([-20,20])
	#ax.set_ylim([6,12])
	ax.minorticks_on()
	ax.axhline(y=6.6, linestyle='--')
	'''
	ax.text(0.145, 0.935,'$P_{dyn} = 2.0 [nPa]$',
	horizontalalignment = 'center',
	verticalalignment = 'center',
	transform = ax.transAxes,
	fontsize = 14)

	verts = [
	(-29.0, 11.40), # left, bottom
	(-29.0, 11.85), # left, top
	(-18.0, 11.85), # right, top
	(-18.0, 11.40), # right, bottom
	(0.0, 0.0), # ignored
	]

	codes = [Path.MOVETO,
	Path.LINETO,
	Path.LINETO,
	Path.LINETO,
	Path.CLOSEPOLY,
	]

	path = Path(verts, codes)
	patch = patches.PathPatch(path, facecolor='white', lw=1)
	ax.add_patch(patch)
	'''
	for k in range(r[0,:].size):
		ax.plot(Bz, r[:,k])
	fig.savefig('./figures/rvsBz/rvsBz.pdf')
	plt.close()

