# -*- coding: utf-8 -*-
"""
Created on Thursday March 13 2014
Magnetopause Location
@author: Jonathan
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
# Definitions.
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
	#labels = ['$B_{z} = %10.2f [nT]$, $P_{dyn} = %10.2f [nPa]$' % (B,D), 'GOES 8 x-y Trajectory $[R_{E}]$', 'GOES 8 x-y Position $[R_{E}]$']
	labels  = ['$B_{z} = %10.2f [nT]$, $P_{dyn} = %10.2f [nPa]$' % (B,D), 'GOES 8 Position $[R_{E}]$']
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
	ax.plot(x,-y,color='b')
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
	fig    = plt.figure(figsize=[7.06,6.06])
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

def plotGOESBzvsTime(GOES_Bz,Time):
	fig    = plt.figure(figsize=[7.06,6.06])
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
	fig.savefig('./figures/GOES_BzvsTime/GOES_BzvsTime.pdf')
	plt.close()

#***********************************************************************
# Read Bz, and u data from file. Set the bugs to 0.
#***********************************************************************
Bz_raw = np.genfromtxt('./omni_min_26813.lst.txt', delimiter="", usecols = (4), dtype=float, autostrip=True)
for i in range(Bz_raw.size):
	if Bz_raw[i] > 2000:
		Bz_raw[i] = np.sum(Bz_raw[i-1] + Bz_raw[i+1])/2
	else:
		Bz_raw[i] = Bz_raw[i]

Dp_raw = np.genfromtxt('./omni_min_26813.lst.txt', delimiter="", usecols = (5), dtype=float, autostrip=True)
for i in range(Dp_raw.size):
	if Dp_raw[i] > 50:
		Dp_raw[i] = np.sum(Dp_raw[i-1] + Dp_raw[i+1])/2
		if Dp_raw[i+1] > 50:
			Dp_raw[i] = np.sum(Dp_raw[i-1] + Dp_raw[i+2])/2
			if Dp_raw[i+2] > 50:
				Dp_raw[i] = np.sum(Dp_raw[i-1] + Dp_raw[i+3])/2
	else:
		Dp_raw[i] = Dp_raw[i]

GOES_Bz_raw = np.genfromtxt('./GOES_POS.txt', delimiter="", usecols = (4), dtype=float, autostrip=True, skiprows=4)
for i in range(GOES_Bz_raw.size):
	if GOES_Bz_raw[i] < -200:
		GOES_Bz_raw[i] = np.sum(GOES_Bz_raw[i-1] + GOES_Bz_raw[i+1])/2
		if GOES_Bz_raw[i+1] < -200:
			GOES_Bz_raw[i] = np.sum(GOES_Bz_raw[i-1] + GOES_Bz_raw[i+2])/2
	else:
		GOES_Bz_raw[i] = GOES_Bz_raw[i]

GOES_X_raw = np.genfromtxt('./GOES_POS.txt', delimiter="", usecols = (5), dtype=float, autostrip=True, skiprows=4)
GOES_Y_raw = np.genfromtxt('./GOES_POS.txt', delimiter="", usecols = (6), dtype=float, autostrip=True, skiprows=4)

#Date = np.genfromtxt('./GOES_POS.txt', delimiter="", usecols = (0), dtype=None, autostrip=True, skiprows=4)
#Date = np.array([datetime.strptime(s, '%d-%m-%Y').date() for s in Date.view('S10')])

#Time = np.genfromtxt('./GOES_POS.txt', delimiter="", usecols = (1), dtype=None, autostrip=True, skiprows=4)
#Time = np.array([datetime.strptime(s, '%H:%M:%S.%f').time() for s in Time.view('S12')])

#d = np.array([datetime.combine(Date, Time) for s in d])

Time_raw = np.genfromtxt('./GOES_POS.txt', delimiter="  ", usecols = (0), dtype=None, autostrip=True, skiprows=4)
Time = np.empty(Time_raw.size/10, dtype='S23')
Time[0] = Time_raw[5]
for i in range(1,Time.size):
	Time[i] = Time_raw[(10*i+5)]
Time = np.array([datetime.strptime(s, '%d-%m-%Y %H:%M:%S.%f') for s in Time.view('S23')])

#***********************************************************************
# Take a 10 minute average of the magnetic field measured by 
# the spacecraft to smooth out the data. Do the same for position 
# and time of the soacecraft. Do the same for the IMF. Basically 
# this will make the data-model comarison easier to comprehend. 
#***********************************************************************
Bz = np.zeros(Bz_raw.size/10)
Bz[0] = np.sum(Bz_raw[0:10])/10
Bz[287] = np.sum(Bz_raw[2870:2879])/10
for i in range(1, Bz.size-1):
	Bz[i] = np.sum(Bz_raw[10*i:10*i+10])/10

Dp = np.zeros(Dp_raw.size/10)
Dp[0] = np.sum(Dp_raw[0:10])/10
Dp[287] = np.sum(Dp_raw[2870:2879])/10
for i in range(1, Dp.size-1):
	Dp[i] = np.sum(Dp_raw[10*i:10*i+10])/10

GOES_Bz = np.zeros(GOES_Bz_raw.size/10)
GOES_Bz[0] = np.sum(GOES_Bz_raw[0:10])/10
GOES_Bz[287] = np.sum(GOES_Bz_raw[2870:2879])/10
for i in range(1, GOES_Bz.size-1):
	GOES_Bz[i] = np.sum(GOES_Bz_raw[10*i:10*i+10])/10

GOES_X = np.zeros(GOES_X_raw.size/10)
GOES_X[0] = np.sum(GOES_X_raw[0:10])/10
GOES_X[287] = np.sum(GOES_X_raw[2870:2879])/10
for i in range(1, GOES_X.size-1):
	GOES_X[i] = np.sum(GOES_X_raw[10*i:10*i+10])/10

GOES_Y = np.zeros(GOES_Y_raw.size/10)
GOES_Y[0] = np.sum(GOES_Y_raw[0:10])/10
GOES_Y[287] = np.sum(GOES_Y_raw[2870:2879])/10
for i in range(1, GOES_Y.size-1):
	GOES_Y[i] = np.sum(GOES_Y_raw[10*i:10*i+10])/10

POS = np.sqrt(GOES_X**2 + GOES_Y**2)/6378.1

#***********************************************************************
# Declaring variables for Shue et al. 1998.
#***********************************************************************
nmax      = Dp.size
r0        = np.zeros(Dp.size)
alpha     = np.zeros(Dp.size)
theta     = np.linspace(0, np.pi, 50)
theta[49] = theta[49] - 4.0e-2

r         = np.zeros([theta.size,Dp.size])

#***********************************************************************
# Calculating the Shue model
#***********************************************************************
for i in range(nmax):
	r0[i]    = (10.22 + 1.29*np.tanh(0.184*(Bz[i] + 8.14)))*(Dp[i])**(-1/6.6)
	alpha[i] = (0.58 - 0.007*Bz[i])*(1 + 0.024*np.log(Dp[i]))

for i in range(nmax):
	r[:,i] = r0[i]*(2/(1 + np.cos(theta[:])))**(alpha[i])

#***********************************************************************
# Print values useful for your contingency table.
#***********************************************************************
GOES_Bz_min = np.min(GOES_Bz)
for i in range(len(GOES_Bz)):
	if GOES_Bz[i] < GOES_Bz_min + 0.001:
		#print i
		Measured_Crossing = i
		break

Measured_Cross_Time = Time[Measured_Crossing]
print "The minimum Bz value measured by GOES is %s [nT]" % GOES_Bz_min
print "The measured initial crossing occurrs in frame %i" % Measured_Crossing
print "The measured initial crossing happened at %s [GMT]" % Measured_Cross_Time

for j in range(len(r)):
	for i in range(len(POS)):
		if r[j,i]<POS[i]:
			#print i
			Predicted_Crossing = i - 1
			break

Predicted_Cross_Time = Time[Predicted_Crossing]
print "The predicted initial crossing occurrs in frame %i" % Predicted_Crossing
print "The predicted initial crossing happened at %s [GMT]" % Predicted_Cross_Time

#***********************************************************************
# Start building the contingency table.
#***********************************************************************
Data_Table = GOES_Bz<0
Model_Table = np.zeros([len(theta),len(POS)])
Model_Table = r<POS

data_magnetosheath = np.where(Data_Table)
model_magnetosheath = np.where(Model_Table[True])

print "Times/Locations that the data shows the satellite in the magnetosheath: %s" % data_magnetosheath
print "Times/Locations that the model shows the satellite in the magnetosheath: %s" % model_magnetosheath

print "Locations where they agree: %s" % model_magnetosheath[0][Data_Table[model_magnetosheath]]
print "Total numbers of agreement: %s" % model_magnetosheath[0][Data_Table[model_magnetosheath]].size
print "Locations where they do not agree: %s" % model_magnetosheath[0][np.logical_not(Data_Table[model_magnetosheath])]
print "Total numbers of disagreement: %s" % model_magnetosheath[0][np.logical_not(Data_Table[model_magnetosheath])].size

#***********************************************************************
# Convert to Cartesian for plotting.
#***********************************************************************
x = np.zeros([theta.size, Dp.size])
y = np.zeros([theta.size, Dp.size])
for i in range(nmax):
	x[:,i] = r[:,i]*np.cos(theta[:])
	y[:,i] = r[:,i]*np.sin(theta[:])

#***********************************************************************
# Plot Stuffs
#***********************************************************************
#***********************************************************************
# Produce many plots for making a movie.
#***********************************************************************
#for i in range(nmax):
#	plotmovie(x[:,i],y[:,i],Bz[i],Dp[i],GOES_X[i],GOES_Y[i],GOES_X[:],GOES_Y[:], Time, i)
	#************************* Print percent done **************************
#	print '%i percent complete' % np.int(i * 100 / nmax)

#***********************************************************************
# Make colored happy plots for the model sensitivity study.
#***********************************************************************
#plotstuffs(x[:,:],y[:,:], Bz[:], Dp[:], GOES_X[:], GOES_Y[:])

#***********************************************************************
# Plot r vs. Dp.
#***********************************************************************
#plotrvsDp(r[0,:,:], Dp[:])

#***********************************************************************
# Plot r vs. Bz.
#***********************************************************************
#plotrvsBz(r[0,:], Bz[:])
#plotrvsBz(r[0,:,:], Bz[:])

#***********************************************************************
# Plot GOES Bz magnetometer data.
#***********************************************************************
plotGOESBzvsTime(GOES_Bz,Time)

#***********************************************************************
# For adding text to a plot.
#***********************************************************************
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

# Build an array that holds the values for GOES_X that are greater than 3400
#goes_wedge_x = np.array(np.where(GOES_X[:] > 3400))

# Build an array that holds the values for GOES_Y that are between -2500 and 2500
#goes_wedge_y = np.array(np.where((-2500 < GOES_Y[:]) & (GOES_Y[:] < 2500)))
