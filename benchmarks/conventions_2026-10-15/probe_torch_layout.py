"""Compare public interpolate on contiguous NCHW and channels-last inputs."""
import pathlib,json,numpy as np,torch
import torch.nn.functional as F
torch.set_num_threads(1)
root=pathlib.Path.home()/'tmp/aic-variants-scratch/zenresize/oracle-dense'
results=[]
for key,w,h,ow,oh in [('123-ramp',24,16,1,23),('109-texture',18,26,39,22),('21-impulse',38,17,21,11)]:
 x=np.fromfile(root/f'{key}.f32','<f4').reshape(h,w,3)
 t=torch.from_numpy(x).permute(2,0,1)[None]
 for mode in ['nearest','nearest-exact','bicubic']:
  kw={} if mode.startswith('nearest') else dict(align_corners=False,antialias=True)
  a=F.interpolate(t,size=(oh,ow),mode=mode,**kw)
  b=F.interpolate(t.contiguous(),size=(oh,ow),mode=mode,**kw)
  results.append(dict(case=key,mode=mode,max_layout_delta=float((a-b).abs().max()),strides=list(t.stride())))
print(json.dumps(results,indent=2))
