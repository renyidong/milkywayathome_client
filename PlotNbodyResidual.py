#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pylab as lab
import subprocess
import math as ma

input = sys.argv[1];
g = open(input, 'r');
x1 = [];
y1 = [];
bID1 = [];
g.next()
g.next()
g.next()
g.next()
g.next()
for line in g:
 ln = line.split(',');
 x1.append(float(ln[1]));
 y1.append(float(ln[2]));
 bID1.append(int(ln[8]));

x1s = [None]*len(bID1);
y1s = [None]*len(bID1);
minB = len(x1)
print str(len(x1)) + ' ' + str(len(y1)) + ' ' + str(len(bID1))
for i in range(len(x1)):
	x1s[bID1[i]] = x1[i];
	y1s[bID1[i]] = y1[i];


plt.subplot(221)
plt.plot(x1s,y1s, 'ob', label="GPU Data")
plt.xlim([-5,5])
plt.ylim([-5,5])
legend = plt.legend(loc="upper right")



input = sys.argv[2];
f = open(input, 'r');
x2 = [];
y2 = [];
bID2 = [];

f.next()
f.next()
f.next()
f.next()
f.next()
for line in f:
 ln = line.split(',');
 x2.append(float(ln[1]));
 y2.append(float(ln[2]));
 bID2.append(int(ln[8]));

x2s = [None]*len(bID2);
y2s = [None]*len(bID2);
for i in range(len(x2)):
	x2s[bID2[i]] = x2[i];
	y2s[bID2[i]] = y2[i];

print "Number of bodies in x list: " + str(len(x2))
print "Number of bodies in y list: " + str(len(y2))

plt.subplot(222)
plt.plot(x2,y2, 'ob', label="CPU Data");
plt.xlim([-5,5])
plt.ylim([-5,5])
legend = plt.legend(loc="upper right")
x3 = []
y3 = []
for i in range(len(x2s)):
	x3.append(x2s[i]-x1s[i])
	y3.append(y2s[i]-y1s[i])

plt.subplot(223)
plt.plot(x3,y3, 'ob', label="Residual");
legend = plt.legend(loc="upper right")

r = []
for i in range(len(x3)):
	r.append(ma.sqrt((x3[i]*x3[i])+(y3[i]*y3[i])))
sum = 0;
for elem in r:
	sum += elem;

averageResidual = sum/(len(r)*1.0)
print "AVERAGE RESIDUAL: " + str(averageResidual) + "\n"
plt.subplot(224)
binwidth = (max(r)-min(r))/50.0
if(binwidth < .001):
	binwidth = .001
plt.hist(r, bins=np.arange(min(r), max(r) + binwidth, binwidth), label="Residual");
legend = plt.legend(loc="upper right")
plt.show();