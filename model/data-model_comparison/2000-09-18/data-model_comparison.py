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
import plotstuffs as ps
from matplotlib.patches import Wedge
from matplotlib.path import Path
from datetime import datetime
from scipy.interpolate import interp1d

reload(ps)

#***********************************************************************
# Read Bz, Dp, and GOES data from file.
# Interpolate the bugs.
#***********************************************************************
Bz_raw = np.genfromtxt('./omni_min_3562.lst.txt', delimiter="", usecols = (4), dtype=float, autostrip=True)
time_temp = np.linspace(0,len(Bz_raw),len(Bz_raw))
T_Good, D_Good = time_temp[Bz_raw != 9999.99], Bz_raw[Bz_raw != 9999.99]
LocBad = Bz_raw == 9999.99
T_Bad = time_temp[LocBad]
func_interp = interp1d(T_Good, D_Good, kind='cubic')
Bz_raw[LocBad] = func_interp(T_Bad)

Dp_raw = np.genfromtxt('./omni_min_3562.lst.txt', delimiter="", usecols = (5), dtype=float, autostrip=True)
time_temp = np.linspace(0,len(Dp_raw),len(Dp_raw))
T_Good, D_Good = time_temp[Dp_raw != 99.99], Dp_raw[Dp_raw != 99.99]
LocBad = Dp_raw == 99.99
T_Bad = time_temp[LocBad]
func_interp = interp1d(T_Good, D_Good, kind='cubic')
Dp_raw[LocBad] = func_interp(T_Bad)

GOES_Bz_raw = np.genfromtxt('./GOES_POS.txt', delimiter="", usecols = (4), dtype=float, autostrip=True, skiprows=4)
time_temp = np.linspace(0,len(GOES_Bz_raw),len(GOES_Bz_raw))
T_Good, D_Good = time_temp[GOES_Bz_raw != -1.0e31], GOES_Bz_raw[GOES_Bz_raw != -1.0e31]
LocBad = GOES_Bz_raw == -1.0e31
T_Bad = time_temp[LocBad]
func_interp = interp1d(T_Good, D_Good, kind='cubic')
GOES_Bz_raw[LocBad] = func_interp(T_Bad)

GOES_X_raw = np.genfromtxt('./GOES_POS.txt', delimiter="", usecols = (5), dtype=float, autostrip=True, skiprows=4)
GOES_Y_raw = np.genfromtxt('./GOES_POS.txt', delimiter="", usecols = (6), dtype=float, autostrip=True, skiprows=4)

Time_raw = np.genfromtxt('./GOES_POS.txt', delimiter="  ", usecols = (0), dtype=None, autostrip=True, skiprows=4)
Time = np.empty(Time_raw.size/10, dtype='S23')
Time[0] = Time_raw[5]
for i in range(1,Time.size):
	Time[i] = Time_raw[(10*i+5)]
Time = np.array([datetime.strptime(s, '%d-%m-%Y %H:%M:%S.%f') for s in Time.view('S23')])
Time_raw = np.array([datetime.strptime(s, '%d-%m-%Y %H:%M:%S.%f') for s in Time_raw.view('S23')])

#***********************************************************************
# Take a 10 minute average of the magnetic field measured by
# the spacecraft to smooth out the data. Do the same for GOES_Rition
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

GOES_R = np.sqrt(GOES_X**2 + GOES_Y**2)/6378.1

#***********************************************************************
# Declaring variables for Shue et al. 1998.
#***********************************************************************
nmax      = Dp.size
r0        = np.zeros(Dp.size)
alpha     = np.zeros(Dp.size)
theta     = np.linspace(0, 2*np.pi, 100)
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
# Convert to Cartesian for plotting.
#***********************************************************************
x = np.zeros([theta.size, Dp.size])
y = np.zeros([theta.size, Dp.size])
for i in range(nmax):
	x[:,i] = r[:,i]*np.cos(theta[:])
	y[:,i] = r[:,i]*np.sin(theta[:])

#***********************************************************************
# Find where the data says the magnetopause crossing is.
# Save the result in an array of 1's and 2's.
#***********************************************************************
measured_crossings = np.zeros(len(GOES_Bz))
for i in range(len(GOES_Bz)):
	if GOES_Bz[i] < 0:
		#print i
		measured_crossings[i] = 2
	else:
		measured_crossings[i] = 1

#***********************************************************************
# Find "theta_GOES" - the angle the spacecraft is at in the same
# "theta" coordiante used to define the magnetopause in the Shue model.
#
# Using theta_GOES, find where the model says the magnetopause
# crossing is. Save the result in an array of 10's and 100's.
#***********************************************************************
theta_GOES = np.zeros(len(GOES_Y))
for i in range(len(GOES_Y)):
	if GOES_Y[i] > 0:
		theta_GOES[i] = 2*np.pi - np.arccos((GOES_X[i]/6378.1)/GOES_R[i])
	else:
		theta_GOES[i] = np.arccos((GOES_X[i]/6378.1)/GOES_R[i])

predicted_crossings = np.zeros(len(GOES_R))
for i in range(len(GOES_R)):
	this_theta = theta[(theta[:] > theta_GOES[i] - .05) & (theta[:] < theta_GOES[i] + .05)]
	this_r = r[(theta[:] > theta_GOES[i] - .05) & (theta[:] < theta_GOES[i] + .05),i]
	if this_r[0]<GOES_R[i]:
		#print i
		predicted_crossings[i] = 100
	else:
		predicted_crossings[i] = 10

#***********************************************************************
# To build the contingency table. Multiply measured_crossings
# with predicted_crossings. Values in the result will give one of
# four unique numbers. Each of which corresponds to a hit, miss,
# false alarm, or correct negative. In the contingency table.
# Store the total number of hits, misses, fa's, and cn's and
# print them.
#***********************************************************************
contingency_table = measured_crossings*predicted_crossings

hit  = len(contingency_table[contingency_table == 200])
miss = len(contingency_table[contingency_table == 20])
fa   = len(contingency_table[contingency_table == 100])
cn   = len(contingency_table[contingency_table == 10])

print 'Total number of hits: %s' % hit
print 'Total number of miss: %s' % miss
print 'Total number of false alarms: %s' % fa
print 'Total number of correct negatives: %s' % cn

#***********************************************************************
# Plot Stuffs
#***********************************************************************
#***********************************************************************
# Plot GOES Bz magnetometer data
#***********************************************************************
ps.plotGOESBzvsTime(GOES_Bz, Time, 'GOES_BzvsTime')
ps.plotGOESBzvsTime(GOES_Bz_raw, Time_raw, 'GOES_BzvsTime_raw')

#***********************************************************************
# Plot Time vs theta_GOES
#***********************************************************************
ps.plotTimevstheta_GOES(Time, theta_GOES)

#***********************************************************************
# Produce many plots for making a movie.
#***********************************************************************
for i in range(nmax):
	ps.plotmovie(x[:,i],y[:,i],Bz[i],Dp[i],GOES_X[i],GOES_Y[i],GOES_X[:],GOES_Y[:], Time, i)
	#************************* Print percent done **************************
	print '%i percent complete' % np.int(i * 100 / nmax)

#***********************************************************************
# Unused junk
#***********************************************************************
'''
#GOES_outside = np.where(GOES_Bz < 0)
#GOES_inside  = np.where(GOES_Bz > 0)
#Measured_Cross_Time = Time[Measured_Crossing]
#print "The minimum Bz value measured by GOES is %s [nT]" % GOES_Bz_min
#print "The measured initial crossing occurrs in frame %i" % Measured_Crossing
#print "The measured initial crossing happened at %s [GMT]" % Measured_Cross_Time

#Predicted_Cross_Time = Time[Predicted_Crossing]
#print "The predicted initial crossing occurrs in frame %i" % Predicted_Crossing
#print "The predicted initial crossing happened at %s [GMT]" % Predicted_Cross_Time

data_outside = GOES_Bz<0
data_inside  = GOES_Bz>0

model_outside = r<GOES_R

data_magnetosheath = np.where(data_outside)
print "Data: Outside Magnetopause %s" % data_magnetosheath[0].size

model_magnetosheath = np.where(model_outside[True])
print "Model: Outside Magnetopause %s" % model_magnetosheath[0].size

agree_magnetosheath = model_magnetosheath[0][data_outside[model_magnetosheath]]
disagree_magnetosheath = model_magnetosheath[0][np.logical_not(data_outside[model_magnetosheath])]

print "Total numbers of agreement: %s" % (agree_magnetosheath.size)
print "Total numbers of disagreement: %s" % (disagree_magnetosheath.size)


#model_outside = np.zeros([len(theta),len(GOES_R)])
#model_inside  = r>GOES_R

#print "Times/Locations that the data shows the satellite in the magnetosheath: %s" % data_magnetosheath
#print "Times/Locations that the model shows the satellite in the magnetosheath: %s" % model_magnetosheath

#data_magnetosphere = np.where(data_inside)
#print "Data: Inside Magnetopause %s" % data_magnetosphere[0].size

#model_magnetosphere = np.where(model_inside[True])
#print "Model: Inside Magnetopause %s" % model_magnetosphere[0].size

#print "Times/Locations that the data shows the satellite in the magnetosphere: %s" % data_magnetosphere
#print "Times/Locations that the model shows the satellite in the magnetosphere: %s" % model_magnetosphere

#agree_magnetosphere = model_magnetosphere[0][data_inside[model_magnetosphere]]
#disagree_magnetosphere = model_magnetosphere[0][np.logical_not(data_inside[model_magnetosphere])]

#print "Locations where they agree: %s" % agree_magnetosphere + agree_magnetosheath
#print "Total numbers of agreement: %s" % (agree_magnetosphere.size + agree_magnetosheath.size)

#print "Locations where they do not agree: %s" % disagree
#print "Total numbers of disagreement: %s" % (disagree_magnetosphere.size + disagree_magnetosheath.size)
'''