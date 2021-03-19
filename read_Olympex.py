from netCDF4 import *
from numpy import *
import datetime

import glob

fstKu=glob.glob("../monthly/SEAsiaCS/2A*DPR*HDF5")

qdataL=[]

f='Atlantic/2A.GPM.DPR.V8-20180723.20180830-S200745-E214019.025594.V06A.HDF5'
f='Atlantic/2A.GPM.DPR.V8-20180723.20180830-S214020-E231254.025595.V06A.HDF5'

def readDPR(f,lat1,lat2):
    f=f.replace("Ku","DPR")
    fh=Dataset(f)
    lat=fh['NS/Latitude'][:,24]
    lon=fh['NS/Longitude'][:,24]
    a=nonzero((lat-lat1)*(lat-lat2)<0)
    n1=a[0][0]
    n2=a[0][-1]
    n1=4600
    n2=4850
    #n1=2200
    #n2=3100
    rrate3d=zeros((n2-n1+1,24,176),float)
    rratesfc=zeros((n2-n1+1,24),float)
    zKu=fh['NS/PRE/zFactorMeasured'][n1:n2,:,:]
    zKa=fh['MS/PRE/zFactorMeasured'][n1:n2,:,:]
    lon=fh['NS/Longitude'][n1:n2,:]
    lat=fh['NS/Latitude'][n1:n2,:]
    bzd=fh['NS/VER']['binZeroDeg'][n1:n2,:]
    bcf=fh['NS/PRE']['binClutterFreeBottom'][n1:n2,:]
    zKu=ma.array(zKu,mask=zKu<10)
    zKa=ma.array(zKa,mask=zKa<10)
    bsfc=fh['NS/PRE/binRealSurface'][n1:n2,:]
    bst=fh['NS/PRE/binStormTop'][n1:n2,:]
    pType=(fh['NS/CSF/typePrecip'][n1:n2,:]/1e7).astype(int)
    dprsfcRate=fh['NS/SLV/precipRateESurface'][n1:n2,:]
    piaSRT=fh['/NS/SRT/pathAtten'][n1:n2,:]
    reliabFlag=fh['NS/SRT/reliabFlag'][n1:n2,:]
    binBB=fh['NS/CSF/binBBPeak'][n1:n2,:]
    binBBT=fh['NS/CSF/binBBTop'][n1:n2,:]
    h0=fh['NS/VER/heightZeroDeg'][n1:n2,:]
    htop=fh['NS/PRE/heightStormTop'][n1:n2,:]
    epsDPR=fh['NS/SLV/epsilon'][n1:n2,:,:]
    sfcType=fh['NS/PRE/landSurfaceType'][n1:n2,:]
    xlon=fh['NS/Longitude'][n1:n2,:]
    ylat=fh['NS/Latitude'][n1:n2,:]
    return xlon,ylat,dprsfcRate,zKu,zKa,pType, fh, n1,n2, bcf,fh
lon1=-73
lon2=-70
import matplotlib.pyplot as plt
fname='2A.GPM.DPR.V8-20180723.20140611-S171129-E184401.001619.V06A.HDF5'
fname='../DPRData/2A.GPM.DPR.V8-20180723.20151203-S142802-E160036.010019.V06A.HDF5'
xlon,ylat,dprsfcRate,zKu,zKa,pType,fh,n1,n2,bcf,fh=readDPR(fname,33.,35)
import matplotlib
#plt.pcolormesh(xlon,ylat,dprsfcRate,cmap='jet',\
#               norm=matplotlib.colors.LogNorm(),vmin=0.1)
#plt.title('Orbit 25595\n30 August 2018')
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
j=22
plt.figure()
plt.pcolormesh(zKu[:,j,:].T,vmax=50,cmap='jet',vmin=10)
plt.plot(arange(250)+0.5,bcf[:,j])
plt.ylim(175,100)
plt.xlabel('Scan Number')
plt.ylabel('Range bin')
cb=plt.colorbar()
cb.ax.set_title('dBZ')
plt.title('Olympex 3 December 2015\nObserved Z(Ku)')
plt.savefig('olympex_2015_12_03.png')
import pickle
d=pickle.load(open("biasCorrectTablesZ.pklz","rb"))
zShape=d["DPR"]["biasD"]
binSfc=fh["NS/PRE/binRealSurface"][n1:n2,:]
bzd=fh["NS/VER/binZeroDeg"][n1:n2,:]
#{"CMB":d1,"DPR":d2},
ielev=(fh['NS/PRE/elevation'][n1:n2,:]/125).astype(int)
zKuC=zKu.copy()
itopL=[]
iSfcL=[]
for i in range(bzd.shape[0]):
    j0=int(bzd[i,j]-128)
    itop=bcf[i,j]-130
    #print(itop,binSfc[i,j]-130)
    itopL.append(itop)
    ibott=binSfc[i,j]-130
    if ielev[i,j]>0:
        ibott=175-ielev[i,j]-130
    iSfcL.append(ibott)
    fzClass=bzd[i,j]-128
    if itop<39 and pType[i,j]>0:
        print(itop,ibott)
        zCS=zShape[itop:min(ibott,39),fzClass]
        #print(zCS)
        if zCS[0]>0.1:
            zKuC[i,j,itop+130:min(ibott,39)+130]=zKuC[i,j,itop+130]/zCS[0]*zCS
            zKuC[i,j,min(ibott,39)+130:176]=nan
    if pType[i,j]>0 and itop>=39:
        print(bcf[i,j])
    if bcf[i,j]>=168:
        zKuC[i,j,bcf[i,j]:]=nan
    if pType[i,j]<1:
        zKuC[i,j,bcf[i,j]:]=nan

#plt.show()

plt.figure()
plt.pcolormesh(zKuC[:,j,:].T,vmax=50,cmap='jet',vmin=10)
#plt.plot(arange(250)+0.5,bcf[:,j])
plt.ylim(175,100)
plt.xlabel('Scan Number')
plt.ylabel('Range bin')
cb=plt.colorbar()
cb.ax.set_title('dBZ')
plt.title('Olympex 3 December 2015\nClutter corrected Z(Ku)')
plt.savefig('olympex_2015_12_03_cluttCorr.png')

i0=120
zKuC_2=zKu.copy()

plt.figure()
plt.pcolormesh(zKu[i0,:,:].T,cmap='jet',vmin=10,vmax=50)
plt.plot(arange(49)+0.5,bcf[i0,:])
plt.ylim(175,100)
plt.xlabel('Ray Number')
plt.ylabel('Range bin')
cb=plt.colorbar()
cb.ax.set_title('dBZ')
plt.title('Olympex 3 December 2015\Observed Z(Ku)')
plt.savefig('olympex_2015_12_03_CrossSect.png')

i=i0
for j in range(49):
    j0=int(bzd[i,j]-128)
    itop=bcf[i,j]-130
    #print(itop,binSfc[i,j]-130)
    itopL.append(itop)
    ibott=binSfc[i,j]-130
    if ielev[i,j]>0:
        ibott=175-ielev[i,j]-130
    iSfcL.append(ibott)
    fzClass=bzd[i,j]-128
    if itop<39 and pType[i,j]>0:
        print(itop,ibott)
        zCS=zShape[itop:min(ibott,39),fzClass]
        #print(zCS)
        if zCS[0]>0.1:
            zKuC_2[i,j,itop+130:min(ibott,39)+130]=zKuC_2[i,j,itop+130]/zCS[0]*zCS
            zKuC_2[i,j,min(ibott,39)+130:176]=nan
    if pType[i,j]>0 and itop>=39:
        print(bcf[i,j])
    if bcf[i,j]>=168:
        zKuC_2[i,j,bcf[i,j]:]=nan
    if pType[i,j]<1:
        zKuC_2[i,j,bcf[i,j]:]=nan

plt.figure()
plt.pcolormesh(zKuC_2[i0,:,:].T,cmap='jet',vmin=10,vmax=50)
plt.ylim(175,100)
plt.xlabel('Ray Number')
plt.ylabel('Range bin')
cb=plt.colorbar()
cb.ax.set_title('dBZ')
plt.title('Olympex 3 December 2015\nClutter corrected Z(Ku)')
plt.savefig('olympex_2015_12_03_CrossSect_CluttCorr.png')
