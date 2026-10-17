"""Produce exact Pillow RGB unit-test pins through its public resize API."""
from PIL import Image, __version__
import numpy as np
print('Pillow',__version__)
for w,h,ow,oh in [(7,5,3,2),(16,12,23,9)]:
 i=np.arange(w*h*3,dtype=np.uint32)
 x=((i*73+i//7*19)%256).astype(np.uint8).reshape(h,w,3)
 data=np.asarray(Image.fromarray(x).resize((ow,oh),Image.Resampling.BICUBIC)).tobytes()
 checksum=14695981039346656037
 for byte in data:checksum=((checksum^byte)*1099511628211)&((1<<64)-1)
 print((w,h,ow,oh),'bytes',list(data) if len(data)<30 else len(data),'fnv1a64',hex(checksum))
