import sys

try:
  import pycuda.gpuarray as gpuarray
  import pycuda.autoinit
except:
  sys.stderr.write("INFO: no pycuda libs\n")

try:
  import numpy as np
  import scikits.cuda.linalg as cu
except:
  sys.stderr.write("INFO: no scikits libs\n")

try:
  useCUDA=True
  sys.stderr.write("INFO: Found CUDA libraries\n")
  cu.init()
except:
  useCUDA=False
  sys.stderr.write("INFO: CUDA not supported\n")

from standalone import uses
debug,info,mprint = uses('debug','info','mprint')

import timeit

def dot(x,y):
    # debug("CUDA dot product")

    cu.init()
    # a = np.asarray(np.random.rand(1300, 8200), np.float32)
    # b = np.asarray(np.random.rand(8200, 1300), np.float32)
    # a_gpu = gpuarray.to_gpu(a)
    # b_gpu = gpuarray.to_gpu(b)
    # c_gpu = cu.dot(a_gpu, b_gpu)
    # mprint("c_gpu",c_gpu.get())
    # assert(np.allclose(np.dot(a, b), c_gpu.get()))

    d = np.asarray(np.random.rand(5), np.float32)
    e = np.asarray(np.random.rand(5), np.float32)
    d_gpu = gpuarray.to_gpu(d)
    e_gpu = gpuarray.to_gpu(e)
    f = cu.dot(d_gpu, e_gpu)
    assert(np.allclose(np.dot(d, e), f))

    a = np.asarray(x, np.float32)
    b = np.asarray(y, np.float32)
    mprint("a",a)
    mprint("b",b)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = cu.dot(a_gpu, b_gpu)
    res = c_gpu.get()
    mprint("cu.dot",c_gpu.get())
    # print(res.shape)
    npdot = np.dot(a,b)
    mprint("np.dot",npdot)
    mprint("a",a)
    mprint("b",b)
    mprint("a_gpu",a_gpu.get())
    mprint("b_gpu",b_gpu.get())
    assert np.allclose(res,npdot),"DARN"
    return res
