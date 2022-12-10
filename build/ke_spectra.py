from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def rm(x,N):
  return np.convolve(x, np.ones(N), 'valid') / N


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
    

avg_size = 21

t = -1

nc = Dataset("orig.nc", "r")
u = nc.variables["uvel"][t,:,:,:]
v = nc.variables["vvel"][t,:,:,:]
w = nc.variables["wvel"][t,:,:,:]
freq,spd = ke_spectra(u,v,w)
spd = rm(spd,avg_size)
freq = freq[int((avg_size-1)/2):int((avg_size-1)/2+len(spd))]
plt.loglog( freq , spd )

print(len(spd))
print(len(rm(spd,9)))

nc = Dataset("sgs.nc", "r")
u = nc.variables["uvel"][t,:,:,:]
v = nc.variables["vvel"][t,:,:,:]
w = nc.variables["wvel"][t,:,:,:]
freq,spd = ke_spectra(u,v,w)
spd = rm(spd,avg_size)
freq = freq[int((avg_size-1)/2):int((avg_size-1)/2+len(spd))]
plt.loglog( freq , spd )

# npzfile = np.load('kh_test_hi_ke_spectra.npz')
# freq_hi = npzfile['freq']
# avg_size_hi = npzfile['avg_size']
# spd_hi = npzfile['spd']
# plt.loglog( rm(freq_hi,avg_size_hi) , rm(spd_hi,avg_size_hi) )

plt.loglog( rm(freq,avg_size) , 1e-2 * rm(freq,avg_size)**(-5./3.) )

plt.legend(["orig","sgs","k^{-5/3}"])

#plt.xlim([6.e-2,freq[-1]])
#plt.ylim()

# example = 1.e-1 * freq**(-3.)
# plt.loglog( freq , example )

plt.show()

