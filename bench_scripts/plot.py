#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys

with open(sys.argv[1]) as f:
  data = f.read()

data = data.split('\n')

del data[-1]

print data

x     = [row.split(',')[0] for row in data]
rep   = data[1].split(',')[0]
data1 = [row.split(',')[2] for row in data]
data2 = [row.split(',')[3] for row in data]
data3 = [row.split(',')[4] for row in data]
data4 = [row.split(',')[5] for row in data]
 
fig = plt.figure()

ax1 = fig.add_subplot(111)

title=sys.argv[2]
ax1.set_title(title)    
ax1.set_xlabel('DOFs')
ax1.set_ylabel('time (s)')

ax1.plot(x,data1, c='r', label='STD')
ax1.plot(x,data2, c='b', label='LO')
ax1.plot(x,data3, c='g', label='ESi')
ax1.plot(x,data4, c='y', label='ESm')

leg = ax1.legend(loc='lower right')

plt.savefig(title+'.png')