import sys
import numpy as np

try:
  import pycuda.gpuarray as gpuarray
  import pycuda.autoinit
except:
  sys.stderr.write("INFO: no pycuda libs\n")

try:
  import scikits.cuda.linalg as linalg
except:
  sys.stderr.write("INFO: no scikits libs\n")

try:
  useCUDA=True
  sys.stderr.write("INFO: Found CUDA libraries\n")
  linalg.init()
except:
  useCUDA=False
  sys.stderr.write("INFO: CUDA not supported\n")

from standalone import uses
debug,info,mprint = uses('debug','info','mprint')

import timeit

def dot(a,b):
    debug("CUDA dot product")
    a_gpu = gpuarray.to_gpu(np.copy(a))
    b_gpu = gpuarray.to_gpu(np.copy(b))
    d_gpu = linalg.dot(a_gpu, b_gpu)

    res = np.asarray(d_gpu.get())
    # print(res.shape)
    return res
