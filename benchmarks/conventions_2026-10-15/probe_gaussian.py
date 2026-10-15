"""Measure TensorFlow's public Gaussian resize operation on impulses."""
import json, os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_NUM_INTEROP_THREADS']='1'
os.environ['TF_NUM_INTRAOP_THREADS']='1'
import numpy as np
import tensorflow as tf
out=[]
for n,m in [(9,31),(11,13),(17,7),(7,1),(8,3),(1,9)]:
    basis=np.eye(n,dtype=np.float32).reshape(1,n,n)
    # Each channel is one source impulse; output exposes every kernel weight.
    for aa in [False,True]:
        weights=tf.image.resize(basis,[1,m],method='gaussian',antialias=aa).numpy()[0]
        out.append(dict(n=n,m=m,antialias=aa,weights=weights.tolist()))
print(json.dumps(dict(version=tf.__version__,probes=out)))
