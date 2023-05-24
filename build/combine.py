import xarray
import sys
if (len(sys.argv) < 2) :
  print("ERROR: Must pass one parameter: output prefix"); sys.exit(-1);
ds = xarray.open_mfdataset(sys.argv[1]+"_*.nc")
ds.to_netcdf(sys.argv[1]+"_combined.nc")
ds.close()
