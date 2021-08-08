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
from datetime import datetime
from scipy.interpolate import interp1d

#***********************************************************************
# Plot Stuffs!
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

def plotmovie(x,y,B,D,GOES_X,GOES_Y,GX,GY, Time, i):
	labels  = ['$B_{z} = %10.2f [nT]$, $P_{dyn} = %10.2f [nPa]$' % (B,D), 'GOES 10 Position $[R_{E}]$']
	fig     = plt.figure(figsize=[7.06,7.06])
	ax      = fig.add_subplot(111)
	ax.set_title('Magnetopause Location \n CME Event September 23 1999')
	ax.set_xlabel('$x$ $[R_{E}]$')
	ax.set_ylabel('$y$ $[R_{E}]$')
	ax.set_xlim([15,-45])
	ax.set_ylim([30,-30])
	ax.minorticks_on()
	dual_half_circle((0, 0), radius=1.0, angle=90, ax=ax)

	ax.text(0.2, 0.03,'%s' % Time[i],
	horizontalalignment = 'center',
	verticalalignment = 'center',
	transform = ax.transAxes,
	fontsize = 14)

	ax.plot(x,y,color='b')
	ax.plot(GOES_X/6378.1,GOES_Y/6378.1, 'o', color='g')
	#ax.plot(x,-y,color='b')
	#ax.plot(GOES_X/6378.1,GOES_Y/6378.1, 'o')
	#for j in range(GX.size):
	#	ax.plot(GX/6378.1,GY/6378.1)
	ax.legend(labels, bbox_to_anchor=(0.02, 0.98), loc=2, borderaxespad=0., fancybox=True, shadow=True)
	fig.savefig('./figures/movie/%s.pdf' % `i` )
	plt.close()

def plotstuffs(x,y,Bz,Dp,GOES_X,GOES_Y):
	labels = []
	for l in range(Bz.size):
		labels.append('$B_{z} = %10.2f [nT]$, $P_{dyn} = %10.2f [nPa]$' % (Bz[l],Dp[l]))
	fig    = plt.figure(figsize=[7.06,7.06])
	ax     = fig.add_subplot(111)
	ax.set_title('Magnetopause Location \n Sensitivity to $P_{dyn}$')
	#ax.set_title('Magnetopause Location \n Sensitivity to $B_{z}$')
	ax.set_xlabel('$x$ $[R_{E}]$')
	ax.set_ylabel('$R$ $[R_{E}]$')
	ax.set_xlim([15,-20])
	ax.set_ylim([0,35])
	ax.minorticks_on()
	dual_half_circle((0, 0), radius=1.0, angle=90, ax=ax)
	#ax.plot(GOES_X,GOES_Y, 'o')
	#for i in range(GOES_X.size):
	#	ax.plot(GOES_X/6378.1,GOES_Y/6378.1)
	for j in range(x[0,:].size):
		ax.plot(x[:,j], y[:,j])
	ax.legend(labels, bbox_to_anchor=(0.02, 0.98), loc=2, borderaxespad=0., fancybox=True, shadow=True)
	#fig.savefig('./figures/dDp.pdf')
	fig.savefig('./figures/dBz.pdf')
	plt.close()

def plotrvsDp(r,Dp):
	labels = ['$B_{z} =  [nPa]$']
	fig    = plt.figure(figsize=[7.06,7.06])
	ax     = fig.add_subplot(111)
	ax.set_title('Magnetopause Location \n Sensitivity to $P_{dyn}$')
	ax.set_xlabel('$P_{dyn} [nPa]$')
	ax.set_ylabel('$r [R_{E}]$')
	#ax.set_xlim([15,-20])
	#ax.set_ylim([0,35])
	ax.minorticks_on()
	for k in range(r[0,:].size):
		ax.plot(Dp, r[:,k])
	#ax.legend(labels, bbox_to_anchor=(0.69, 0.98), loc=2, borderaxespad=0., fancybox=True, shadow=True)
	fig.savefig('./figures/rvsDp/rvsDp.pdf')
	plt.close()

def plotrvsBz(r,Bz):
	#labels = ['$r [R_{E}]$', '$B_{z} [nT]$']
	fig    = plt.figure(figsize=[7.06,7.06])
	ax     = fig.add_subplot(111)
	ax.set_title('Magnetopause Location \n Sensitivity to $B_{z}$')
	ax.set_xlabel('$B_{z} [nT]$')
	ax.set_ylabel('$r [R_{E}]$')
	#ax.set_xlim([-20,20])
	#ax.set_ylim([6,12])
	ax.minorticks_on()
	#ax.plot(Bz, r)
	for k in range(r[0,:].size):
		ax.plot(Bz, r[:,k])
	fig.savefig('./figures/rvsBz/rvsBz.pdf')
	plt.close()

def plotGOESBzvsTime(GOES_Bz, Time, name):
	fig    = plt.figure(figsize=[7.06,7.06])
	ax     = fig.add_subplot(111)
	ax.set_title('GOES 10 Magnetometer')
	ax.set_xlabel('Time')
	ax.set_ylabel('$B_{z} [nT]$')
	#ax.set_xlim([-200,200])
	ax.set_ylim([-200,250])
	ax.minorticks_on()
	ax.xaxis_date()
	# Make space for and rotate the x-axis tick labels
	fig.autofmt_xdate()
	ax.plot(Time, GOES_Bz)
	fig.savefig('./figures/GOES_BzvsTime/%s.pdf' % name)
	plt.close()

def plotTimevstheta_GOES(Time, theta_GOES):
	fig    = plt.figure(figsize=[7.06,7.06])
	ax     = fig.add_subplot(111)
	#ax.set_title('GOES 10 Magnetometer')
	ax.set_xlabel('Time')
	ax.set_ylabel('$\\theta_{GOES} [rad]$')
	#ax.set_xlim([Time[107],Time[146]])
	#ax.set_ylim([-200,250])
	ax.minorticks_on()
	ax.xaxis_date()
	# Make space for and rotate the x-axis tick labels
	fig.autofmt_xdate()
	ax.plot(Time, theta_GOES)
	fig.savefig('./figures/Timevstheta_GOES.pdf')
	plt.close()

