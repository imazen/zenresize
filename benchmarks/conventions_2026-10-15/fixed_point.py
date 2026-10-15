"""Independent arithmetic feasibility models, never library implementation code.

Candidate coefficients are rounded to a power-of-two integer scale. OpenCV
uses the requested documented 11-bit scale and one final 22-bit rounding;
Pillow candidates use two separately rounded/clipped u8 passes. Bit depths
are experimental controls, not claims about Pillow implementation internals.
"""
import json,math,pathlib,sys
import numpy as np

def kernel(x,c):
 t=abs(x)
 if c is None:return max(0.,1-t)
 return 1+t*t*(-3+c+(2-c)*t) if t<1 else c*(4+t*(-8+t*(5-t))) if t<2 else 0.
def plan(n,m,c,aa,bits,edge,rounding):
 scale=m/n;shrink=min(scale,1) if aa else 1
 result=[]
 for i in range(m):
  center=(i+.5)/scale-.5;radius=(1 if c is None else 2)/shrink
  taps={}
  for j in range(math.ceil(center-radius),math.floor(center+radius)+1):
   weight=kernel((j-center)*shrink,c)
   if edge=='truncate' and not 0<=j<n:continue
   index=min(n-1,max(0,j));taps[index]=taps.get(index,0.)+weight
  total=sum(taps.values());q=1<<bits
  def quant(x):
   if rounding=='even':return round(x)
   if rounding=='positive':return math.floor(x+.5)
   return math.floor(x+.5) if x>=0 else -math.floor(-x+.5)
  result.append([(j,quant(w/total*q)) for j,w in taps.items() if w!=0])
 return result

def resize(x,ow,oh,c,aa,bits,edge,two_pass,rounding,final_round='up'):
 h,w,ch=x.shape;hp=plan(w,ow,c,aa,bits,edge,rounding);vp=plan(h,oh,c,aa,bits,edge,rounding)
 horizontal=np.zeros((h,ow,ch),np.int64)
 for i,taps in enumerate(hp):
  for j,q in taps:horizontal[:,i]+=x[:,j].astype(np.int64)*q
 if two_pass:horizontal=np.clip((horizontal+(1<<(bits-1)))>>bits,0,255)
 output=np.zeros((oh,ow,ch),np.int64)
 for i,taps in enumerate(vp):
  for j,q in taps:output[i]+=horizontal[j]*q
 shift=bits if two_pass else 2*bits
 if final_round=='even':
  whole=output>>shift;remainder=output-(whole<<shift);half=1<<(shift-1)
  rounded=whole+((remainder>half)|((remainder==half)&((whole&1)!=0)))
 else:rounded=(output+(1<<(shift-1)))>>shift
 return np.clip(rounded,0,255).astype(np.uint8)

if __name__ == '__main__':
 root=pathlib.Path.home()/'tmp/aic-variants-scratch/zenresize'/sys.argv[1]
 records=json.loads((root/'results.json').read_text())['cases']
 summaries=[]
 for framework,c,aa,edge,two_pass in [('cv-cubic',.75,False,'clamp',False),('cv-linear',None,False,'clamp',False),('pillow-cubic',.5,True,'truncate',True)]:
  for bits in ([11] if framework.startswith('cv') else [11,14,16,22]):
   for rounding in ['even','positive','away']:
    worst=0;different=0;pixels=0;cases=0;worst_case=''
    for row in records:
     name=framework+'-u8'
     if row['convention']!=name:continue
     w,h,ow,oh=row['shape'];key=row['case']
     x=np.fromfile(root/f'{key}.u8',np.uint8).reshape(h,w,3)
     expected=np.fromfile(root/f'{key}-{name}-oracle.u8',np.uint8).reshape(oh,ow,3)
     got=resize(x,ow,oh,c,aa,bits,edge,two_pass,rounding)
     delta=np.abs(got.astype(np.int16)-expected.astype(np.int16));maximum=int(delta.max())
     if maximum>worst:worst=maximum;worst_case=key
     different+=int(np.any(delta!=0,axis=2).sum());pixels+=ow*oh;cases+=1
    summaries.append(dict(framework=framework,bits=bits,rounding=rounding,cases=cases,pixels=pixels,max_u8=worst,differing=different,differing_pct=100*different/pixels,worst_case=worst_case))
 print(json.dumps(summaries,indent=2))
 (root/'fixed-point-expanded-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
