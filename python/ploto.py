import matplotlib.pyplot as pl
import numpy as np

f = open("out.dat","r")

line = f.readline()
y = [float(yi) for yi in line.split()]

f.close()        

x = np.linspace(0,2*np.pi,20)
z = [np.sin(xi)+np.random.rand()/5 for xi in x]

pl.plot(x,z,'s-')
pl.plot(x,y,'.-')
pl.show()
