#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys

with open(sys.argv[1]) as f:
  data = f.read()

data = data.split('\n')

del data[0]
del data[-1]
del data[-1]

x     = [row.split(',')[0] for row in data]
rep   = [row.split(',')[1] for row in data]
data1 = [row.split(',')[2] for row in data]
data2 = [row.split(',')[3] for row in data]

# print data1
# print data2
fig = plt.figure()

ax1 = fig.add_subplot(111)
title=sys.argv[2]
ax1.set_title(title)    
ax1.set_xlabel('DOFs')
ax1.set_xscale('log')
ax1.set_ylabel('time (s)')
num_max=int(sys.argv[3])
# print num_max
plt.ylim(ymax=num_max)

ax1.plot(x,data1, c='r', label='RAW')
ax1.plot(x,data2, c='b', label='LO')
 
leg = ax1.legend(loc='lower right')

plt.savefig(title+'.png')
