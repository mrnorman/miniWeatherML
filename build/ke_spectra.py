from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def rm(x,N):
  ret = x
  h = int(np.floor(N/2))
  nx = x.shape[0]
  for i in range(nx) :
    if (i-h<0) :
      ret[i] = np.average(x[0:i+1])
    elif (i-h+N>nx-1) :
      ret[i] = np.average(x[i:nx])
    else :
      ret[i] = np.average(x[i-h:i-h+N])
  return ret


def ke_spectra(u,v,w) :
  spd = np.abs( np.fft.rfft2(u[:,0,:]) )**2
  spd = 0
  for j in range(u.shape[1]) :
    ke = (u[:,j,:]*u[:,j,:] + v[:,j,:]*v[:,j,:] + w[:,j,:]*w[:,j,:]) / 2
    spd = spd + np.abs( np.fft.rfft2(ke) )**2
  freq = np.fft.rfftfreq(len(ke[0,:]))
  freq2d = np.sqrt(np.outer(freq,freq))
  spd /= u.shape[1]
  spd = spd.reshape(spd.shape[0]*spd.shape[1])
  freq2d = freq2d.reshape(freq2d.shape[0]*freq2d.shape[1])
  indices = np.argsort(freq2d)
  freq2d = freq2d[indices[:]]
  spd    = spd   [indices[:]]
  # import sys
  # import numpy
  # numpy.set_printoptions(threshold=sys.maxsize)
  L = [[freq2d[i],spd[i]] for i in range(len(freq2d))]
  L1 = pd.DataFrame(L).groupby(0, as_index=False)[1].mean().values.tolist()
  freq2d = np.asarray([L1[i][0] for i in range(len(L1))])
  spd    = np.asarray([L1[i][1] for i in range(len(L1))])
  return freq2d , spd
    

nc = Dataset("test.nc", "r")
u = nc.variables["uvel"][-1,:,:,:]
v = nc.variables["vvel"][-1,:,:,:]
w = nc.variables["wvel"][-1,:,:,:]
freq,spd = ke_spectra(u,v,w)

plt.loglog( freq , rm(spd,20) )

example = 1.e-1 * freq**(-5./3.)
plt.loglog( freq , example )

example = 2.e-4 * freq[int(len(freq)*1/2):]**(-6.)
plt.loglog( freq[int(len(freq)*1/2):] , example )

#plt.xlim([4.e-2,freq[-1]])
#plt.ylim()

# example = 1.e-1 * freq**(-3.)
# plt.loglog( freq , example )

plt.show()

