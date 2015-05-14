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

import timeit

def dot(a,b):
    # print("Pump to GPU\n")
    # start_time1 = timeit.default_timer()
    # print(a.shape)
    # print(b.shape)
    a_gpu = gpuarray.to_gpu(np.copy(a))
    b_gpu = gpuarray.to_gpu(np.copy(b))
    # print("Time elapsed ",timeit.default_timer() - start_time1)

    # print("Calculate\n")
    # start_time2 = timeit.default_timer()
    d_gpu = linalg.dot(a_gpu, b_gpu)
    # print(d_gpu)
    # print(d_gpu.shape)
    # print("Time elapsed ",timeit.default_timer() - start_time1)

    res = np.asarray(d_gpu.get())
    # print(res.shape)
    return res
