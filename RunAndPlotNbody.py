#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pylab as lab
import subprocess

timesteps = 1000

os.system('rm GPUBRUTE.out')
os.system('rm CPUBRUTE.out')
os.system('rm GPUACCTEST.out')
os.system('rm CPUACCTEST.out')

print "RUNNING CPU SYSTEM:"
os.system("cmake -DNBODY_OPENCL=OFF -DDOUBLEPREC=OFF -DNBODY_OPENMP=OFF -DCMAKE_BUILD_TYPE=RELEASE")
os.system('make')
executeString = './bin/milkyway_nbody -f nbody/sample_workunits/EMD_10k_plummer.lua -o CPUBRUTE.out -z CPU.hist -x -i -e 36912 ' + str(timesteps) + ' 1 .2 12' #>> CPUACCTEST.out'
os.system(executeString)

print "RUNNING GPU SYSTEM:"
os.system("cmake -DNBODY_OPENCL=ON -DDOUBLEPREC=OFF -DCMAKE_BUILD_TYPE=RELEASE ")
os.system('make')
executeString = './bin/milkyway_nbody -f nbody/sample_workunits/EMD_10k_plummer.lua -o GPUBRUTE.out -z GPU.hist -x -i -e 36912 ' + str(timesteps) + ' 1 .2 12' #>> GPUACCTEST.out'
os.system(executeString)


print "==========================FILE DIFF===============================\n"
os.system('diff CPUACCTEST.out GPUACCTEST.out')
print "==================================================================\n"
os.system('./bin/milkyway_nbody -h CPU.hist -s GPU.hist')
print "PLOTTING DATA"
os.system('python2 PlotNbodyResidual.py GPUBRUTE.out CPUBRUTE.out')