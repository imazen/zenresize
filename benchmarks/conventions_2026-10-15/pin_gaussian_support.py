"""Pin hard support masks from TensorFlow's public Gaussian resize API."""
import os
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2',TF_ENABLE_ONEDNN_OPTS='0',TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='1')
import numpy as np,tensorflow as tf
for n,m in [(18,39),(26,22),(29,19),(14,29),(32,30),(5,32)]:
 for aa in [False,True]:
  oracle=tf.image.resize(np.eye(n,dtype=np.float32)[None,:,:],[1,m],method='gaussian',antialias=aa).numpy()[0]
  inv=np.float32(n)/np.float32(m);shrink=np.float32(1)/max(inv,np.float32(1)) if aa else np.float32(1)
  for i,row in enumerate(oracle):
   dist=(np.arange(n,dtype=np.float32)+np.float32(.5)-np.float32(i+.5)*inv)*shrink
   wrong=np.abs(dist)<np.float32(1.5)
   if np.any(wrong!=(row>0)):
    print(n,m,aa,i,'support',np.nonzero(row>0)[0].tolist(),'direct ratio support',np.nonzero(wrong)[0].tolist())
