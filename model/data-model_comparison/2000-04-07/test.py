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
# Hey, JNick.

# Here's a way easier idea.  You have a time array, "T", and a data array, 
# "data".  data has an arbitrary number of bad data points.  It looks like 
# you're trying to interpolate using the good points to fill in the bad points. 
# That makes sense, but many nested "if" statements is not a good way to do it 
# because it is not arbitrarily robust- you are basically defining the depth 
# to which you'll search for the nearest good neighbors.  If you "sound out" 
# what you're doing (spell it out in English) and do so for an arbitrary case, 
# you'll see there are several solutions.  However, this would be the easiest:

time = np.linspace(0, 100, 100)
x = np.linspace(0, 2*np.pi, 100)
data = np.sin(x)
data[10] = 99.99
data[30] = 99.99
data[45] = 99.99
data[90] = 99.99

plt.plot(time, data)
plt.show()

# First, separate the wheat from the chaffe:
T_Good, D_Good = time[data != 99.99], data[data != 99.99] # where baddat is your bad data flag.

# Now, find out what times correspond to bad points:
LocBad = data == 99.99
T_Bad = time[LocBad]

# Use scipy's interpolation stuff to interpolate from good points to bad points!  I'm too lazy to look this up, but it works something like this:
func_interp = interp1d(T_Good, D_Good, kind='cubic') # something like that.

#Note that you may need to turn datetimes into numbers, use matplotlib.dates.date2num or whatever.

# Use interp function to fill in bad spots.
data[LocBad] = func_interp(T_Bad)

plt.plot(time, data)
plt.show()

