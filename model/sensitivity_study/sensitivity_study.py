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

reload(ps)

#***********************************************************************
# rvsDp
#***********************************************************************
Bz = np.linspace(-60, -10, 50)
Dp = np.linspace(5, 18, 50)

#***********************************************************************
# rvsBz
#***********************************************************************
#Bz = np.linspace(-30, 15, 50)
#Dp = np.linspace(9, 34, 50)

#Bz = np.linspace(-30, 30, 5)
#Dp = np.linspace(1.0, 1.0, 5)

nmax = Dp.size

#***********************************************************************
# Declaring variables for Shue et al. 1998.
#***********************************************************************
r0    = np.zeros(Dp.size)
alpha = np.zeros(Dp.size)
theta = np.linspace(0, np.pi, 50)
theta[49] = theta[49] - 4.0e-2

r = np.zeros([theta.size,Dp.size,Bz.size])

#***********************************************************************
# Begin the loop.
#***********************************************************************
for k in range(len(Bz)):
	#***********************************************************************
	# Calculating the Shue model
	#***********************************************************************
	for i in range(nmax):
		r0[i]    = (10.22 + 1.29*np.tanh(0.184*(Bz[k] + 8.14)))*(Dp[i])**(-1/6.6)
		alpha[i] = (0.58 - 0.007*Bz[k])*(1 + 0.024*np.log(Dp[i]))

	for i in range(nmax):
		r[:,i,k] = r0[i]*(2/(1 + np.cos(theta[:])))**(alpha[i])

	#***********************************************************************
	# Convert to Cartesian for plotting.
	#***********************************************************************
	x = np.zeros([theta.size, Dp.size])
	y = np.zeros([theta.size, Dp.size])
	for i in range(nmax):
		x[:,i] = r[:,i,k]*np.cos(theta[:])
		y[:,i] = r[:,i,k]*np.sin(theta[:])

#***********************************************************************
# Plot Stuffs
#***********************************************************************
#***********************************************************************
# Make colored happy plots for the model sensitivity study.
#***********************************************************************
#ps.plotstuffs(x[:,:], y[:,:], Bz[:], Dp[:])

#***********************************************************************
# Plot r vs. Dp.
#***********************************************************************
ps.plotrvsDp(r[0,:,:], Dp[:])

#***********************************************************************
# Plot r vs. Bz.
#***********************************************************************
#ps.plotrvsBz(r[0,:,:], Bz[:])
