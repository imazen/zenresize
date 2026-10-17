"""Black-box convention comparison; no oracle implementation source is used."""
import os
os.environ.update(OPENCV_IPP='disabled',TF_CPP_MIN_LOG_LEVEL='2',TF_ENABLE_ONEDNN_OPTS='0',TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='1')
import json, pathlib, subprocess, sys
import numpy as np
import cv2
import PIL
from PIL import Image
import torch
import torch.nn.functional as F
import tensorflow as tf
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)
cv2.ipp.setUseIPP_NotExact(False)
assert not cv2.ipp.useIPP()
torch.set_num_threads(1)
torch.set_grad_enabled(False)
root=pathlib.Path.home()/'tmp/aic-variants-scratch/zenresize'/sys.argv[2]
root.mkdir(exist_ok=False)
binary=str(pathlib.Path(sys.argv[1]).resolve())
summary={}
records=[]
shapes=[(1,1,3,5),(1,7,9,3),(9,1,3,7),(7,5,3,2),(5,3,11,7),(13,17,9,8),(16,12,23,9),(8,6,8,6),(8,4,3,3)]
if '--dense' in sys.argv:
 rng=np.random.default_rng(20261015)
 shapes.extend(tuple(int(x) for x in row) for row in rng.integers(1,41,size=(128,4)))
for case,(w,h,ow,oh) in enumerate(shapes):
 for pattern in ['texture','impulse','ramp','fractional']:
  index=np.arange(w*h*3,dtype=np.uint32).reshape(h,w,3)
  if pattern in ['texture','fractional']:x=((index*73+index//7*19)%256).astype(np.uint8)
  elif pattern=='ramp':x=((index*255)//max(1,w*h*3-1)).astype(np.uint8)
  else:
   x=np.zeros((h,w,3),np.uint8);x[h//2,w//2]=[255,129,17]
  key=f'{case}-{pattern}'
  x.tofile(root/f'{key}.u8')
  floating=x.astype(np.float32)
  if pattern=='fractional':floating/=255.0
  floating.astype('<f4').tofile(root/f'{key}.f32')
  tensor=torch.from_numpy(floating).permute(2,0,1)[None]
  def torch_resize(mode,align=False,aa=False):
   options={} if mode.startswith('nearest') else dict(align_corners=align,antialias=aa)
   return F.interpolate(tensor,size=(oh,ow),mode=mode,**options)[0].permute(1,2,0).numpy()
  oracles=[
   ('cv-nearest-f32','nearest-cv',cv2.resize(floating,(ow,oh),interpolation=cv2.INTER_NEAREST)),
   ('cv-cubic-f32','cubic075',cv2.resize(floating,(ow,oh),interpolation=cv2.INTER_CUBIC)),
   ('cv-linear-f32','linear',cv2.resize(floating,(ow,oh),interpolation=cv2.INTER_LINEAR)),
   ('torch-cubic-f32','cubic075',torch_resize('bicubic')),
   ('torch-cubic-align-f32','cubic075-align',torch_resize('bicubic',True)),
   ('torch-linear-f32','linear',torch_resize('bilinear')),
   ('torch-linear-align-f32','linear-align',torch_resize('bilinear',True)),
   ('torch-cubic-aa-f32','cubic05-aa',torch_resize('bicubic',aa=True)),
   ('torch-nearest-f32','nearest-torch',torch_resize('nearest')),
   ('torch-nearest-exact-f32','nearest-half',torch_resize('nearest-exact')),
   ('cv-cubic-u8','cubic075-u8',cv2.resize(x,(ow,oh),interpolation=cv2.INTER_CUBIC)),
   ('cv-linear-u8','linear-u8',cv2.resize(x,(ow,oh),interpolation=cv2.INTER_LINEAR)),
   ('pillow22-cubic-u8','pillow22-u8',np.asarray(Image.fromarray(x).resize((ow,oh),Image.Resampling.BICUBIC))),
   ('pillow-cubic-u8','cubic05-aa-u8',np.asarray(Image.fromarray(x).resize((ow,oh),Image.Resampling.BICUBIC))),
   ('tf-gaussian-f32','gaussian',tf.image.resize(floating,[oh,ow],method='gaussian',antialias=False).numpy()),
   ('tf-gaussian-aa-f32','gaussian-aa',tf.image.resize(floating,[oh,ow],method='gaussian',antialias=True).numpy()),
  ]
  cache={}
  for name,mode,expected in oracles:
   is_u8=mode.endswith('-u8');suffix='u8' if is_u8 else 'f32'
   expected.astype('u1' if is_u8 else '<f4').tofile(root/f'{key}-{name}-oracle.{suffix}')
   if mode not in cache:
    output=root/f'{key}-{mode}-native.{suffix}'
    subprocess.run([binary,mode,str(w),str(h),str(ow),str(oh),str(root/f'{key}.{suffix}'),str(output)],check=True)
    cache[mode]=np.fromfile(output,dtype='u1' if is_u8 else '<f4').reshape(oh,ow,3)
   got=cache[mode];assert np.isfinite(got).all() and np.isfinite(expected).all()
   error=np.abs(got.astype(np.float64)-expected.astype(np.float64))
   byte_scale=255.0 if pattern=='fractional' and not is_u8 else 1.0
   rounded=lambda a:np.clip(np.rint(a.astype(np.float64)*byte_scale),0,255).astype(np.int16)
   error8=np.abs(rounded(got)-rounded(expected))
   row=dict(convention=name,case=key,shape=[w,h,ow,oh],max_abs=float(error.max()),pixels=oh*ow,differing=int(np.any(error!=0,axis=2).sum()),max_u8=int(error8.max()),differing_u8=int(np.any(error8!=0,axis=2).sum()))
   records.append(row)
   d=summary.setdefault(name,dict(cases=0,pixels=0,differing=0,max_abs=0.,max_u8=0,differing_u8=0))
   for k in ['pixels','differing','differing_u8']:d[k]+=row[k]
   d['cases']+=1;d['max_abs']=max(d['max_abs'],row['max_abs']);d['max_u8']=max(d['max_u8'],row['max_u8'])
  print('case',key,flush=True)
for d in summary.values():
 d['differing_pct']=100*d['differing']/d['pixels'];d['differing_u8_pct']=100*d['differing_u8']/d['pixels']
metadata=dict(torch=torch.__version__,tensorflow=tf.__version__,opencv=cv2.__version__,pillow=PIL.__version__,numpy=np.__version__,ipp=cv2.ipp.useIPP(),tf_onednn=os.environ['TF_ENABLE_ONEDNN_OPTS'],threads=1)
(root/'results.json').write_text(json.dumps(dict(metadata=metadata,summary=summary,cases=records),indent=2)+'\n')
print(json.dumps(summary,indent=2))
