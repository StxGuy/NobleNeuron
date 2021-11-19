import matplotlib.pyplot as pl
import numpy as np

x = np.linspace(0,2*np.pi,20)
y = [np.sin(xi) for xi in x]

pl.plot(x,y,'.-')
pl.show()
