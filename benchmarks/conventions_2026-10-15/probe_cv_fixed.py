"""Independent 11-bit models: coefficient precision and partition-of-unity.

No oracle source is consulted. The last cubic coefficient can equivalently
be derived from the mathematical identity that the four cubic weights sum to
one. This tests both that identity and direct evaluation, in f32 and f64.
"""
import json,math,pathlib,sys
import numpy as np
import fixed_point as model
root=pathlib.Path.home()/'tmp/aic-variants-scratch/zenresize'/sys.argv[1]
records=json.loads((root/'results.json').read_text())['cases']
output=[]
for precision in ['f32','f64']:
 for partition in [False,True]:
  def plan(n,m,c,aa,bits,edge,rounding):
   result=[]
   cast=np.float32 if precision=='f32' else float
   for i in range(m):
    center=(i+.5)*(n/m)-.5;first=math.floor(center);phase=cast(center-first)
    if c is None:
     positions=[first,first+1];weights=[cast(1)-phase,phase]
    else:
     positions=[first-1,first,first+1,first+2]
     weights=[model.kernel(cast(j)-phase,cast(c)) for j in [-1,0,1,2]]
     if partition:weights[-1]=cast(1)-((weights[0]+weights[1])+weights[2])
    folded={}
    for j,w in zip(positions,weights):
     index=min(n-1,max(0,j));folded[index]=folded.get(index,0)+round(float(w*(1<<bits)))
    result.append(list(folded.items()))
   return result
  model.plan=plan
  for final_round in ['up','even']:
   for mode,c in [('cubic',.75),('linear',None)]:
    different=0;pixels=0;worst=0;cases=0
    for row in records:
     if row['convention']!=f'cv-{mode}-u8':continue
     w,h,ow,oh=row['shape'];key=row['case']
     x=np.fromfile(root/f'{key}.u8',np.uint8).reshape(h,w,3)
     expected=np.fromfile(root/f'{key}-cv-{mode}-u8-oracle.u8',np.uint8).reshape(oh,ow,3)
     got=model.resize(x,ow,oh,c,False,11,'clamp',False,'even',final_round=final_round)
     delta=np.abs(got.astype(np.int16)-expected.astype(np.int16))
     different+=int(np.any(delta!=0,axis=2).sum());pixels+=ow*oh;worst=max(worst,int(delta.max()));cases+=1
    output.append(dict(mode=mode,final_round=final_round,precision=precision,partition=partition,cases=cases,pixels=pixels,max_u8=worst,differing_pct=100*different/pixels,differing=different))
print(json.dumps(output,indent=2))
with (root/'cv-fixed-rounding-summary.json').open('x') as output_file:
 output_file.write(json.dumps(output,indent=2)+'\n')
