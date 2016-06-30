#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pylab as lab
import subprocess

input = sys.argv[1];
f = open(input, 'r');
x = [];
y = [];
f.next()
f.next()
f.next()
f.next()
f.next()
for line in f:
 ln = line.split(',');
 x.append(float(ln[1]));
 y.append(float(ln[2]));

plt.subplot(121)
plt.plot(x,y, 'ob', label="GPU Data")
# plt.xlim([-9,9])
# plt.ylim([-9,9])
legend = plt.legend(loc="upper right")



input = sys.argv[2];
f = open(input, 'r');
x = [];
y = [];
f.next()
f.next()
f.next()
f.next()
f.next()
for line in f:
 ln = line.split(',');
 x.append(float(ln[1]));
 y.append(float(ln[2]));

plt.subplot(122)
plt.plot(x,y, 'ob', label="CPU Data");
# plt.xlim([-9,9])
# plt.ylim([-9,9])
legend = plt.legend(loc="upper right")
plt.show();
