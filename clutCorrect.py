from os import walk
#from readGPROFSub import *
mypath='/itedata/ITE057/2015/09'
mypath='/gpmdata/'
f1 = []
f2 = []
f3 = []
f4 = []
import glob
#print glob.glob("/home/adam/*.txt")
import datetime
s=datetime.datetime(2018,9,1)
#import h5py as h5
import numpy as np

from  netCDF4 import Dataset
ifig=0
import glob
import matplotlib.pyplot as plt
fs=glob.glob("../cmbv7Temp/CMBOut/2B*201810*")
fs=sorted(fs)
iplot=0
import pickle
d=pickle.load(open("biasCorrectTablesPRate.pklz","rb"))
pRateShape=d["CMB"]["biasD"]

import xarray as xr
from numba import jit

@jit(nopython=True)
def vprCorrection(precipRate,sfcPrecipRate,pRateShape,bcf,bzd,binSfc,pType):
    nx,ny,nz=precipRate.shape
    s1=0
    s2=0
    s3=0
    sfcPrecipRateC=sfcPrecipRate.copy()
    for i in range(nx):
        for j in range(ny):
            j0=int(bzd[i,j]-128)
            itop=bcf[i,j]*2-130
            ibott=binSfc[i,j]*2-130
            fzClass=bzd[i,j]*2-128
            if itop<39 and pType[i,j]>0:
                #print(sfcPrecipRate[i,j],precipRate[i,j,bcf[i,j]-50])
                s1+=sfcPrecipRate[i,j]
                s2+=precipRate[i,j,bcf[i,j]-50]
                pRateCS=pRateShape[itop:min(ibott,39),fzClass]
                if pRateCS[0]>0.1:
                    cPRate=precipRate[i,j,bcf[i,j]-50]*pRateCS/pRateCS[0]
                    nExt=cPRate.shape[0]
                    kmax=1
                    sfcPrecipRateC[i,j]=precipRate[i,j,bcf[i,j]-50]
                    pRL=precipRate[i,j,bcf[i,j]-50]
                    for k in range(1,binSfc[i,j]-bcf[i,j]+1):
                        if 2*k<nExt:
                            precipRate[i,j,bcf[i,j]-50+k]=cPRate[2*k]
                            pRL=precipRate[i,j,bcf[i,j]-50+k]
                            kmax=k
                    s3+=pRL
                    sfcPrecipRateC[i,j]=pRL
                    #print(kmax,bcf[i,j]*2,binSfc[i,j]*2)
            #zKuC[i,j,itop+130:min(ibott,39)+130]=zKuC[i,j,itop+130]/zCS[0]*zCS
            #zKuC[i,j,min(ibott,39)+130:176]=nan
    #print(s1,s2,s3)
    return sfcPrecipRateC
print(len(fs))
for f in fs[:]:
    cAlg=Dataset(f)
    precipRate=cAlg['NS/precipTotRate'][:,:,50:]
    sfcPrecipRate=cAlg['NS/surfPrecipTotRate'][:,:]
    lon=cAlg['NS/Longitude'][:,:]
    lat=cAlg['NS/Latitude'][:,:]
    bzd=cAlg['NS/Input/zeroDegBin'][:,:]
    bsfc=cAlg['NS/Input/surfaceRangeBin'][:,:]
    bcf=cAlg['NS/Input/lowestClutterFreeBin'][:,:]
    pType=(cAlg['NS/Input/precipitationType'][:,:]/1e7).astype(int)
    n1,n2=0,8000
    sfcPrecipRateC=vprCorrection(precipRate[n1:n2,:,:],sfcPrecipRate[n1:n2,:],pRateShape,bcf[n1:n2,:],\
                      bzd[n1:n2,:],bsfc[n1:n2,:],pType[n1:n2,:])
    fnameout='out/'+f.split('/')[-1][:-4]+'CC'+'.HDF5'
    sfcPrecipCX=xr.DataArray(sfcPrecipRateC)
    sfcPrecipX=xr.DataArray(sfcPrecipRate)
    lonX=xr.DataArray(lon)    
    latX=xr.DataArray(lat)
    dS=xr.Dataset({"sfcPrecip":sfcPrecipX,"sfcPrecipCX":sfcPrecipCX,"lon":lonX,"lat":latX})
    dS.to_netcdf(fnameout)
    if iplot==1:
        plt.pcolormesh(sfcPrecipRate[700:900,:].T,\
                           norm=matplotlib.colors.LogNorm()\
                           ,cmap='jet',vmin=0.1)


    
