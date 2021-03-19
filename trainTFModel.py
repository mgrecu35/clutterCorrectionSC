import pickle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from sklearn.preprocessing import StandardScaler

from numba import jit
import matplotlib.pyplot

matplotlib.rcParams.update({'font.size': 13})

from netCDF4 import Dataset
fh=Dataset("NH_v2L.nc")
pRate=fh["pRate"][:,:]
#pRate=fh["zKuLx"][:,:]
pRate_cmb=fh["pRate_cmb"][:,:]
zKu=fh["zKuLx"][:,:]
zKa=fh["zKaLx"][:,:]
bzdL=fh["bzdL"][:]
bcL=fh["bcL"][:]
nx=pRate.shape[0]
nbins=5
pRate[pRate<0]=0
tData1=np.zeros((nx,nbins+3),float)
#stop
zKu[zKu<0]=0
zKa[zKa<0]=0
bcf_pdf=np.array([120.        , 124.37174267, 131.13458529, 136.66434484,
                  139.98388137, 142.04715574, 143.57709165, 144.83205079,
                  145.86025046, 146.73650951, 147.49823392, 148.18410836,
                  148.79182491, 149.33678357, 149.84904318, 150.30835003,
                  150.74553255, 151.15421207, 151.52319544, 151.8921788 ,
                  152.2222416 , 152.53623599, 152.85023038, 153.13201187,
                  153.38441586, 153.63681984, 153.88922383, 154.11561683,
                  154.32166498, 154.52771313, 154.73376128, 154.93980943,
                  155.10282272, 155.24807695, 155.39333118, 155.53858541,
                  155.68383965, 155.82909388, 155.97434811, 156.10817015,
                  156.23954026, 156.37091036, 156.50228047, 156.63365057,
                  156.76502068, 156.89639078, 157.02119874, 157.12151544,
                  157.22183215, 157.32214885, 157.42246555, 157.52278226,
                  157.62309896, 157.72341567, 157.82373237, 157.92404908,
                  158.02652907, 158.13575231, 158.24497555, 158.35419879,
                  158.46342203, 158.57264527, 158.68186851, 158.79109175,
                  158.90031499, 159.00932946, 159.11616199, 159.22299453,
                  159.32982706, 159.43665959, 159.54349213, 159.65032466,
                  159.7571572 , 159.86398973, 159.97082227, 160.08025852,
                  160.19067309, 160.30108766, 160.41150223, 160.5219168 ,
                  160.63233137, 160.74274594, 160.85316051, 160.96357508,
                  161.09354684, 161.23314658, 161.37274631, 161.51234604,
                  161.65194577, 161.7915455 , 161.93114523, 162.08704866,
                  162.25882016, 162.43059167, 162.60236318, 162.77413468,
                  162.94590619, 163.26286389, 163.64656045, 164.08085722,
                  167.        ])

@jit(nopython=True)
def test_random(zKu,pRate,bzdL,bcL,icv,tData1,nbins,bcf_pdf):
    s=0
    nx=pRate.shape[0]
    ic=0
    for i in range(nx):
        i1r=int(100*np.random.random())
        #print(i1r,bcf_pdf[i1r])
        ir=168-int(bcf_pdf[i1r])
        nbins2=int(nbins/2)
        if zKu[i,68-ir]>0.0 and bcL[i]>=168 and\
           pRate[i,68]>-0.01:
            #if pRate[i,33]<-0.01:
            #    pRate[i,33]=pRate[i,32]
            #if pRate[i,34]<-0.01:
            #    pRate[i,34]=pRate[i,32]
            #tData1[ic,0:nbins2]=zKu[i,68-ir-nbins2+1:68-ir+1]
            #tData1[ic,nbins2:nbins]=zKa[i,68-ir-nbins2+1:68-ir+1]
            tData1[ic,0:nbins]=pRate[i,68-ir-nbins+1:68-ir+1]
            tData1[ic,nbins]=168-ir
            tData1[ic,nbins+1]=bzdL[i]
            tData1[ic,nbins+2]=pRate[i,68]
            ic+=1
    icv[0]=ic
nbins=16
icv=np.zeros((1),int)
nx=zKu.shape[0]
tData1=np.zeros((nx,nbins+3),float)
test_random(zKu,pRate,bzdL,bcL,icv,tData1,nbins,bcf_pdf)
x_train=tData1[:icv[0],:nbins+2].copy()
y_train=tData1[:icv[0],-1:].copy()
scaler = StandardScaler()
scalerY = StandardScaler()
x_train=scaler.fit_transform(x_train)
y_train=scalerY.fit_transform(y_train)

@jit(nopython=True)
def bias(tData1,biasDist,countCF,sumCF,sumS):
    nx=tData1.shape[0]
    for i in range(nx):
        i0=int(tData1[i,-3]-130)
        j0=int(tData1[i,-2]-128)
        if i0>=0 and i0<35 and j0>0 and j0<50:
            countCF[i0,j0]+=1
            sumCF[i0,j0]+=tData1[i,-4]
            sumS[i0,j0]+=tData1[i,-1]

@jit(nopython=True)
def bias2(zKu,pRate,bzdL,bcL,biasD,countCF,sumCF,sumS):
    s=0
    nx=pRate.shape[0]
    ic=0
    for i in range(nx):
        if bcL[i]>=168:
            for j in range(130,min(169,bcL[i])):
                i0=j-130
                j0=int(bzdL[i]-128)
                if i0>=0 and i0<39 and j0>=0 and j0<50:
                    countCF[i0,j0]+=1
                    sumCF[i0,j0]+=pRate[i,j-100]
                    sumS[i0,j0]+=pRate[i,68]
@jit(nopython=True)
def bias2_cmb(zKu,pRate,bzdL,bcL,biasD,countCF,sumCF,sumS):
    s=0
    nx=pRate.shape[0]
    ic=0
    for i in range(nx):
        if bcL[i]>=168 and pRate[i,32]>-0.01:
            for j in range(130,min(169,bcL[i])):
                i0=j-130
                j0=int(bzdL[i]-128)
                if i0>=0 and i0<39 and j0>=0 and j0<50:
                    countCF[i0,j0]+=1
                    j2=int((j-100)/2)
                    if j2==33 and pRate[i,j2]<0:
                        pRate[i,j2]=pRate[i,32]
                    if pRate[i,j2]>0:
                        sumCF[i0,j0]+=pRate[i,j2]
                    if pRate[i,34]<0:
                        pRate[i,34]=pRate[i,32]
                    sumS[i0,j0]+=pRate[i,34]
    

biasD=np.zeros((39,50),float)+1
sumCF=np.zeros((39,50),float)
sumS=np.zeros((39,50),float)
countCF=np.zeros((39,50),float)

#bias2(zKu,pRate,bzdL,bcL,biasD,countCF,sumCF,sumS)
bias2_cmb(zKu,pRate_cmb,bzdL,bcL,biasD,countCF,sumCF,sumS)
a=np.nonzero(countCF>0)
biasD[a]=sumCF[a]
biasD[a]/=sumS[a]

for i in range(50):
    for j in range(1,39,2):
        biasD[j,i]=(biasD[j-1,i]+biasD[j+1,i])/2.0
d1={"biasD":biasD.copy(),"countCF":countCF.copy()}
plt.pcolormesh(np.arange(50)+128,np.arange(39)+130,100*(biasD-1),\
               cmap='YlGnBu_r',vmin=-100,vmax=0)
plt.ylim(168,130)
plt.xlabel('Zero Degree Bin')
plt.ylabel('Lowest Free Clutter Bin')
plt.title('Normalized bias combined algorithm')
c=plt.colorbar()
c.ax.set_title('%')
plt.savefig('mBiasCMB.png')

biasD=np.zeros((39,50),float)+1
sumCF=np.zeros((39,50),float)
sumS=np.zeros((39,50),float)
countCF=np.zeros((39,50),float)


plt.figure()
bias2(zKu,pRate,bzdL,bcL,biasD,countCF,sumCF,sumS)
a=np.nonzero(countCF>0)
biasD[a]=sumCF[a]
biasD[a]/=sumS[a]


plt.pcolormesh(np.arange(50)+128,np.arange(39)+130,100*(biasD-1),\
               cmap='YlGnBu_r',vmin=-100,vmax=0)
plt.ylim(168,130)
plt.xlabel('Zero Degree Bin')
plt.ylabel('Lowest Free Clutter Bin')
plt.title('Normalized bias DPR')
c=plt.colorbar()
c.ax.set_title('%')
plt.savefig('mBiasDPR.png')
d2={"biasD":biasD,"countCF":countCF}
pickle.dump({"CMB":d1,"DPR":d2},open("biasCorrectTablesPRate.pklz","wb"))
stop

x_val=tData1[:icv[0],:nbins+2]
y_val=tData1[:icv[0],-1:]
x_val=scaler.transform(x_val)
y_val=scalerY.transform(y_val)



def dmodel(ndims=13):
    inp = tf.keras.layers.Input(shape=(ndims,))
    out1 = tf.keras.layers.Dense(16,activation='relu')(inp)
    out1 = tf.keras.layers.Dropout(0.1)(out1)
    out2 = tf.keras.layers.Dense(16,activation='relu')(out1)
    out2 = tf.keras.layers.Dropout(0.1)(out2)
    #out3 = tf.keras.layers.Dense(16,activation='relu')(out2)
    #out3 = tf.keras.layers.Dropout(0.1)(out3)
    out = tf.keras.layers.Dense(1)(out2)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model



cModel=dmodel(nbins+2)
cModel.compile(loss='mse', \
               optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])


history = cModel.fit(x_train, y_train, batch_size=64,epochs=25,\
                     validation_data=(x_val, y_val))


cModel.save("zEstimatAll_7.h5")
