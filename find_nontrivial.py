import os,sys
import xarray as xr
import numpy as np
path = '/mnt/shared_b/data/hydro_simulations/data/'
# print(os.listdir(path))

ncfiles = list([])
for file in os.listdir(path):
    if file.endswith(".nc"):
        ncfiles.append(file)
print('Total amount of files:', len(ncfiles))
# ncfiles_all = ncfiles.copy()


filecount = 0
rvcount = 0 
while filecount < len(ncfiles):
    ncfiles_cp = ncfiles.copy()
    filename1 = ncfiles[filecount]
    sim1 = xr.open_dataarray(path+filename1)
    for filename2 in ncfiles_cp[filecount+1:]:
        sim2 = xr.open_dataarray(path+filename2)
        if np.sum(np.abs(sim1.isel(t=-1).values - sim2.isel(t=-1).values)) == 0:
            ncfiles.remove(filename2)
            rvcount += 1
    filecount += 1
    
    if filecount % 10 == 0:
        print('{} processed, {} remaining, {} removed by far'.format(filecount+1,len(ncfiles)-filecount-1,rvcount))
    
print('\nTotal amount of nontrivial files: ', len(ncfiles))

outpath = '/home/huangz78/hydro/usefulfiles.npz'
np.savez(outpath, ncfiles=ncfiles)
print('\nuseful files saved! exit~')