from netCDF4 import Dataset
import glob
import numpy as np
m=[1,2,12]
mm='_DJF'
title='January,February and December 2018'
#mm='_JJA'
#title='July to August 2018'
fsL=[]
for m1 in m:
    fs=glob.glob("out/2B*2018%2.2i*HDF5"%m1)
    fs=sorted(fs)
    fsL.extend(fs)
fs=fsL

dx=2.5
nxg=int(130/2.5)
nyg=int(360/2.5)
sfcPrecipG=np.zeros((nxg,nyg),float)
sfcPrecipC_G=np.zeros((nxg,nyg),float)
countG=np.zeros((nxg,nyg),float)

from numba import jit

@jit(nopython=True)
def gridPrecip(sfcPrecip,sfcPrecipC,sfcPrecipG,sfcPrecipC_G,countG,lon,lat,dx):
    nx,ny=sfcPrecip.shape
    nxg,nyg=sfcPrecipG.shape
    for i in range(nx):
        for j in range(ny):
            i0=int((lat[i,j]+65)/dx)
            if lon[i,j]<0:
                lon[i,j]+=360
            j0=int((lon[i,j])/dx)
            if i0>=0 and j0>=0 and i0<nxg and j0<nyg and sfcPrecip[i,j]==sfcPrecip[i,j]:
                countG[i0,j0]+=1
                sfcPrecipC_G[i0,j0]+=sfcPrecipC[i,j]
                sfcPrecipG[i0,j0]+=sfcPrecip[i,j]

for f in sorted(fs)[:]:
    cF=Dataset(f)
    print(f)
    sfcPrecip=cF["sfcPrecip"][:,:]
    sfcPrecipC=cF["sfcPrecipCX"][:,:]
    lon=cF["lon"][:,:]
    lat=cF["lat"][:,:]
    gridPrecip(sfcPrecip,sfcPrecipC,sfcPrecipG,sfcPrecipC_G,countG,lon,lat,dx)

import matplotlib.pyplot as plt
import matplotlib

import cartopy.crs as ccrs


a=np.nonzero(countG>0)
sfcPrecipG[a]/=countG[a]
sfcPrecipC_G[a]/=countG[a]
rainmap=0
sfcPrecipC_G=np.ma.array(sfcPrecipC_G,mask=sfcPrecipC_G<0.01)
sfcPrecipG=np.ma.array(sfcPrecipG,mask=sfcPrecipG<0.01)

if rainmap==1:
    plt.figure(figsize=(10,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    plt.pcolormesh(np.arange(nyg+1)*dx,np.arange(nxg+1)*dx-65,sfcPrecipG,cmap='jet',transform=ccrs.PlateCarree())
    
    plt.figure(figsize=(10,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    plt.pcolormesh(np.arange(nyg+1)*dx,np.arange(nxg+1)*dx-65,sfcPrecipC_G,cmap='jet',transform=ccrs.PlateCarree())
    
ratio=np.zeros((nxg,nyg),float)
a=np.nonzero(sfcPrecipG>0.001)
ratio[a]=sfcPrecipC_G[a]/sfcPrecipG[a]-1
    
plt.figure(figsize=(10,6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ratio=np.ma.array(ratio,mask=ratio<0.05)
c=plt.pcolormesh(np.arange(nyg+1)*dx+dx/2,np.arange(nxg+1)*dx-65+dx/2,ratio*100,cmap='jet',transform=ccrs.PlateCarree(),\
                     vmin=0,vmax=50)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', alpha=0.5, linestyle='--', draw_labels=True)
plt.title(title)
cbar=plt.colorbar(c,orientation='horizontal')
cbar.ax.set_title('[%]')
plt.savefig('mapDiff_percentage%s.png'%mm)
plt.figure(figsize=(10,6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ratio=np.ma.array(ratio,mask=ratio<0.05)
diffSfc=np.ma.array(sfcPrecipC_G-sfcPrecipG,mask=24*abs(sfcPrecipC_G-sfcPrecipG)<0.1)
c=plt.pcolormesh(np.arange(nyg+1)*dx+dx/2,np.arange(nxg+1)*dx-65+dx/2,diffSfc*24,transform=ccrs.PlateCarree(),\
                   cmap='jet',vmin=0.0,vmax=2.0)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', alpha=0.5, linestyle='--', draw_labels=True)
plt.title(title)
cbar=plt.colorbar(c,orientation='horizontal')
cbar.ax.set_title('mm/day')
plt.savefig('mapDiff_mmPerDay%s.png'%mm)

matplotlib.rcParams.update({'font.size': 13})
plt.figure()
plt.plot(np.arange(nxg)*dx-65+dx/2,(sfcPrecipC_G.mean(axis=-1)/sfcPrecipG.mean(axis=-1)-1)*100)
plt.xlabel('Latitude')
plt.title('Surface precipitation relative difference')
plt.ylabel('(%)')
plt.xlim(-63.75,63.75)
plt.title(title)
plt.savefig('sfcPrecipRelDiff%s.png'%mm)
