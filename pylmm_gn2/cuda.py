import sys
from scipy.linalg.blas import dgemm

try:
  import pycuda.gpuarray as gpuarray
  import pycuda.autoinit
except:
  sys.stderr.write("INFO: no pycuda libs\n")

import scikits.cuda.linalg as cu
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
import lmmoptions
debug,info,mprint = uses('debug','info','mprint')

import timeit

def dot(x,y):
    # debug("CUDA dot product")

    cu.init()
    options = lmmoptions.get()

    if options.debug:
        mprint("x",x)
        mprint("y",y)

    # First use numpy
    npdot = np.dot(x,y)
    if options.debug:
      mprint("np.dot",npdot)
    # A = np.asarray(x, np.float64)
    # B = np.asarray(y, np.float64)
    A = x
    B = y

    # Next use dgemm
    if not A.flags['F_CONTIGUOUS']:
        AA = A.T
        transA = True
        GtransA = 'T'
    else:
        AA = A
        transA = False
        GtransA = 'N'

    if not B.flags['F_CONTIGUOUS']:
        BB = B.T
        transB = True
        GtransB = 'T'
    else:
        BB = B
        transB = False
        GtransB = 'N'

    if options.debug:
        mprint("AA",AA)
        mprint("BB",BB)
 
    res = dgemm(alpha=1.,a=AA,b=BB,trans_a=transA,trans_b=transB)
    assert np.allclose(res,npdot),"Numpy does not match linalg dgemm"

    # Now use GPU
    a = A
    b = B
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)

    def dinfo():
        if options.debug:
            mprint("a",a)
            mprint("b",b)
            print "a_gpu strides",a_gpu.strides
            print "b_gpu strides",b_gpu.strides
            print "a_gpu c flag",a_gpu.flags.c_contiguous
            print "b_gpu c flag",b_gpu.flags.c_contiguous
            a_f_order = a_gpu.strides[1] > a_gpu.strides[0]
            b_f_order = b_gpu.strides[1] > b_gpu.strides[0]
            print "a_f_order",a_f_order
            print "b_f_order",b_f_order
    # If strides are equal you can't tell the order. So we'll use numpy instead
    if a_gpu.strides[1] == a_gpu.strides[0] or b_gpu.strides[1] == b_gpu.strides[0]:
        res = npdot
    else:
        info("Final order")
        if a_f_order and not b_f_order:
            info("Transpose B")
            b = b.T
            b_gpu = gpuarray.to_gpu(b)
            dinfo()
            c_gpu = cu.dot(a_gpu, b_gpu, 'N','T')
        elif b_f_order and not a_f_order:
            info("Transpose A")
            a = a.T
            a_gpu = gpuarray.to_gpu(a)
            dinfo()
            c_gpu = cu.dot(a_gpu, b_gpu, 'T','N')
        else:
            dinfo()
            if a_gpu.flags.c_contiguous:
                info("Double transpose")
                a_gpu = gpuarray.to_gpu(a.T)
                b_gpu = gpuarray.to_gpu(b.T)
                c_gpu = cu.dot(a_gpu, b_gpu, 'T','T')
            else:
                info("No transpose")
                c_gpu = cu.dot(a_gpu, b_gpu, 'N','N')
        # c_gpu = cu.dot(a_gpu, b_gpu, GtransA, GtransB)
        mprint("cu.dot",c_gpu.get())
        res = c_gpu.get()
    assert np.allclose(res,npdot),"GPU does not match numpy"
    # return np.asarray(res)
    return np.asarray(npdot)


