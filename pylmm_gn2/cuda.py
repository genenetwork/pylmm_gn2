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
debug,info,mprint = uses('debug','info','mprint')

import timeit

def dot(x,y):
    # debug("CUDA dot product")

    cu.init()
    # d = np.asarray(np.random.rand(500,100), np.float64)
    # e = np.asarray(np.random.rand(100,500), np.float64)
    # mprint("d",d)
    # mprint("e",e)
    # d_gpu = gpuarray.to_gpu(d)
    # e_gpu = gpuarray.to_gpu(e)
    # f = cu.dot(d_gpu, e_gpu)
    # assert(np.allclose(np.dot(d, e), f.get()))

    mprint("x",x)
    mprint("y",y)
    A = np.asarray(x, np.float64, order='F')
    B = np.asarray(y, np.float64, order='F')
    if not A.flags['F_CONTIGUOUS']:
       AA = A.T
       transA = True
    else:
       AA = A
       transA = False

    if not B.flags['F_CONTIGUOUS']:
       BB = B.T
       transB = True
    else:
       BB = B
       transB = False

    res = dgemm(alpha=1.,a=AA,b=BB,trans_a=transA,trans_b=transB)
    # mprint("cu.dot",c_gpu.get())
    # res = c_gpu.get()
    print(res.shape)
    npdot = np.dot(x,y)
    mprint("np.dot",npdot)
    # mprint("a",a)
    # mprint("b",b)
    # rint("a_gpu",a_gpu.get())
    # rint("b_gpu",b_gpu.get())
    assert np.allclose(res,npdot),"DARN"
    return res

def other():
    mprint("a",a)
    mprint("b",b)
    print "a strides",a.strides
    #print "a C order",a.flags.c_contiguous)
    print "b strides",b.strides
    #print "b C order",b.flags.c_contiguous)
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print "a_gpu strides",a_gpu.strides
    print "b_gpu strides",b_gpu.strides
    a_f_order = a_gpu.strides[1] > a_gpu.strides[0]
    b_f_order = b_gpu.strides[1] > b_gpu.strides[0]
    print "a_f_order",a_f_order
    print "b_f_order",b_f_order

    # transpose = False
    # if a.shape[0] != b.shape[0]:
    #   info("Transposing!")
    #   transpose = True
    #   b = b.T
    # a_gpu = gpuarray.to_gpu(a)
    # b_gpu = gpuarray.to_gpu(b)
    # print "a_gpu strides",a_gpu.strides
    # print "b_gpu strides",b_gpu.strides
    info("Final order")
    if not b_f_order:
        b = b.T
        b_gpu = gpuarray.to_gpu(b)
        mprint("a",a)
        mprint("b",b)
        print "a_gpu strides",a_gpu.strides
        print "b_gpu strides",b_gpu.strides
        a_f_order = a_gpu.strides[1] > a_gpu.strides[0]
        b_f_order = b_gpu.strides[1] > b_gpu.strides[0]
        print "a_f_order",a_f_order
        print "b_f_order",b_f_order
        c_gpu = cu.dot(a_gpu, b_gpu, 'N','T')
    elif not a_f_order:
        a = a.T
        a_gpu = gpuarray.to_gpu(a)
        mprint("a",a)
        mprint("b",b)
        print "a_gpu strides",a_gpu.strides
        print "b_gpu strides",b_gpu.strides
        a_f_order = a_gpu.strides[1] > a_gpu.strides[0]
        b_f_order = b_gpu.strides[1] > b_gpu.strides[0]
        print "a_f_order",a_f_order
        print "b_f_order",b_f_order
        c_gpu = cu.dot(a_gpu, b_gpu, 'T','N')
    else:
        mprint("a",a)
        mprint("b",b)
        print "a_gpu strides",a_gpu.strides
        print "b_gpu strides",b_gpu.strides
        a_f_order = a_gpu.strides[1] > a_gpu.strides[0]
        b_f_order = b_gpu.strides[1] > b_gpu.strides[0]
        print "a_f_order",a_f_order
        print "b_f_order",b_f_order
        c_gpu = cu.dot(a_gpu, b_gpu, 'N','N')
    mprint("cu.dot",c_gpu.get())
    res = c_gpu.get()
    print(res.shape)
    npdot = np.dot(x,y)
    mprint("np.dot",npdot)
    mprint("a",a)
    mprint("b",b)
    mprint("a_gpu",a_gpu.get())
    mprint("b_gpu",b_gpu.get())
    assert np.allclose(res,npdot),"DARN"
    return res

