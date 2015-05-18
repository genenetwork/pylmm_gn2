import sys

try:
  from scipy.linalg.blas import dgemm
except:
  sys.stderr.write("INFO: no scipy.linalg.blas libs\n")

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
import lmmoptions
debug,info,mprint = uses('debug','info','mprint')

import threading
lock = threading.Lock()

cu.init()

def dot(x,y):
    global lock

    options = lmmoptions.get()

    if options.debug:
        debug("enter CUDA dot product")
        mprint("x",x)
        mprint("y",y)
        # First use numpy
        npdot = np.dot(x,y)
        if options.useBLAS:
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

            res = dgemm(alpha=1.,a=AA,b=BB,trans_a=transA,trans_b=transB)
            assert np.allclose(res,npdot),"Numpy does not match linalg dgemm"
            # res.reshape(npdot.shape)
            print "res.strides",res.strides
            print "npdot.strides",npdot.strides
            # assert res.strides==npdot.strides

    # Now use GPU
    a = x
    b = y
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = None
    a_f_order = a_gpu.strides[1] > a_gpu.strides[0]
    b_f_order = b_gpu.strides[1] > b_gpu.strides[0]

    def dinfo():
        if options.debug:
            mprint("a",a)
            mprint("b",b)
            mprint("a_gpu",a_gpu.get())
            a_f_order = a_gpu.strides[1] > a_gpu.strides[0]
            mprint("b_gpu",b_gpu.get())
            b_f_order = b_gpu.strides[1] > b_gpu.strides[0]

            print "x c flag",x.flags.c_contiguous,"\ta_gpu c flag",a_gpu.flags.c_contiguous
            print "x f flag",x.flags.f_contiguous,"\ta_gpu f flag",a_gpu.flags.f_contiguous,"\ta_f_order",a_f_order
            print "y c flag",y.flags.c_contiguous,"\tb_gpu c flag",b_gpu.flags.c_contiguous
            print "y f flag",y.flags.f_contiguous,"\tb_gpu f flag",b_gpu.flags.f_contiguous,"\tb_f_order",b_f_order
            print "a_gpu strides",a_gpu.strides
            print "b_gpu strides",b_gpu.strides
            if a_f_order and not a_gpu.flags.f_contiguous:
                raise Exception("Flags out of order")
            if b_f_order and not b_gpu.flags.f_contiguous:
                raise Exception("Flags out of order")
    res = None
    if lock.locked():
        debug("Waiting for lock")
    # If strides are equal you can't tell the order. So we'll use numpy instead
    if a_gpu.strides[1] == a_gpu.strides[0] or b_gpu.strides[1] == b_gpu.strides[0]:
        if options.debug:
           info("Usign numpy instead of CUDA")
        return np.dot(x,y)
    else:
        if a_f_order and not b_f_order:
            if options.debug:
                info("Transpose B (b_gpu)")
            b = b.T
            b_gpu = gpuarray.to_gpu(b)
            dinfo()
            with lock:
                c_gpu = cu.dot(a_gpu, b_gpu, 'N','T')
                res = c_gpu.get()
        elif b_f_order and not a_f_order:
            if options.debug:
                info("Transpose A (a_gpu)")
            a = a.T
            a_gpu = gpuarray.to_gpu(a)
            dinfo()
            with lock:
                c_gpu = cu.dot(a_gpu, b_gpu, 'T','N')
                res = c_gpu.get()
        else:
            dinfo()
            if a_gpu.flags.c_contiguous:
                info("Double transpose - don't expect this to happen")
                a_gpu = gpuarray.to_gpu(a.T)
                b_gpu = gpuarray.to_gpu(b.T)
                c_gpu = cu.dot(a_gpu, b_gpu, 'T','T')
            else:
                if options.debug:
                    info("No transpose")
                with lock:
                    c_gpu = cu.dot(a_gpu, b_gpu, 'N','N')
                    res = c_gpu.get()
    if options.debug:
        mprint("numpy.dot",npdot)
        res = np.ascontiguousarray(res)
        mprint("cu.dot",res)
        assert np.allclose(res,npdot),"GPU does not match numpy"
        # Update strides
        debug("Note we are updating strides by hand here!")
        # res=np.lib.stride_tricks.as_strided(res, shape=npdot.shape, strides=npdot.strides)
        print "res.shape",res.shape
        print "npdot.shape",npdot.shape
        print "res.strides",res.strides
        print "npdot.strides",npdot.strides
        assert res.shape == npdot.shape
        assert res.strides == npdot.strides
        debug("exit CUDA dot product - all tests pass")
    return res
    # return np.ascontiguousarray(res)
    # return np.asarray(res)


