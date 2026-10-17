"""Fit coordinate arithmetic against public oracle operations, not source."""
import os
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2',TF_ENABLE_ONEDNN_OPTS='0',TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='1',OPENCV_IPP='disabled')
import json,math,numpy as np,cv2,tensorflow as tf
cv2.ipp.setUseIPP(False);cv2.setNumThreads(1)
cv_counts=dict(rational=0,direct64=0,reciprocal64=0,division64=0)
for n in range(1,65):
 for m in range(1,65):
  expected=cv2.resize(np.arange(n,dtype=np.float32)[None,:],(m,1),interpolation=cv2.INTER_NEAREST)[0]
  for i,x in enumerate(expected):
   for name,value in [('rational',i*n//m),('direct64',math.floor(i*(n/m))),('reciprocal64',math.floor(i*(1/(m/n)))),('division64',math.floor(i/(m/n)))]:
    cv_counts[name]+=value!=int(x)
results={}
for reciprocal in [False,True]:
 for divide in [False,True]:
  worst=0.;mask=0;cells=0
  for n,m in [(9,31),(11,13),(17,7),(8,3),(18,39),(26,22),(29,19),(14,29),(32,30),(5,32)]:
   for aa in [False,True]:
    oracle=tf.image.resize(np.eye(n,dtype=np.float32)[None,:,:],[1,m],method='gaussian',antialias=aa).numpy()[0]
    inv=np.float32(1)/np.float32(np.float32(m)/np.float32(n)) if reciprocal else np.float32(n)/np.float32(m)
    kernel_scale=max(inv,np.float32(1)) if aa else np.float32(1)
    for i,row in enumerate(oracle):
     center=np.float32(i+.5)*inv
     distance=(np.arange(n,dtype=np.float32)+np.float32(.5))-center
     distance=distance/kernel_scale if divide else distance*(np.float32(1)/kernel_scale)
     weights=np.where(np.abs(distance)<np.float32(1.5),np.exp(-2*distance.astype(np.float64)**2),0.)
     weights/=weights.sum()
     worst=max(worst,float(np.max(np.abs(row-weights))));mask+=int(np.sum((row==0)!=(weights==0)));cells+=n
  results[f'reciprocal={reciprocal},divide={divide}']=dict(max_abs=worst,support_mismatches=mask,cells=cells)
print(json.dumps(dict(cv_nearest_index_mismatches=cv_counts,tf_gaussian=results),indent=2))
