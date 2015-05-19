import sys

try:
  from scipy.linalg.blas import dgemm
except:
  sys.stderr.write("INFO: no scipy.linalg.blas libs\n")

try:
  import pycuda.autoinit
  import pycuda.gpuarray as gpuarray
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

def dot(x,y):
    options = lmmoptions.get()

    debug("enter CUDA dot product")
    size = x.shape[0]*x.shape[1]+y.shape[0]*y.shape[1]
    fpsize = 8
    # If strides are equal you can't tell the order. So we'll use numpy instead
    if x.strides[1] == x.strides[0] or y.strides[1] == y.strides[0] or size*fpsize<20000:
        debug("Using numpy instead of CUDA")
        return np.dot(x,y)

    a = x
    b = y
    a_f_order = a.flags.f_contiguous
    b_f_order = b.flags.f_contiguous
    debug("CUDA allocate size=%d x %d (%f GB)" % (size,fpsize,(float(size)*fpsize/1000000000.0)))

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
    try:
        if a_f_order and not b_f_order:
            debug("Transpose B (b_gpu)")
            b = b.T
            a_gpu = gpuarray.to_gpu(a)
            b_gpu = gpuarray.to_gpu(b)
            dinfo()
            c_gpu = cu.dot(a_gpu, b_gpu, 'N','T')
            res = c_gpu.get()
        elif b_f_order and not a_f_order:
            debug("Transpose A (a_gpu)")
            a = a.T
            a_gpu = gpuarray.to_gpu(a)
            b_gpu = gpuarray.to_gpu(b)
            dinfo()
            c_gpu = cu.dot(a_gpu, b_gpu, 'T','N')
            res = c_gpu.get()
        else:
            dinfo()
            if a_gpu.flags.c_contiguous:
                raise Exception("Double transpose - don't expect this to happen")
                a_gpu = gpuarray.to_gpu(a.T)
                b_gpu = gpuarray.to_gpu(b.T)
                c_gpu = cu.dot(a_gpu, b_gpu, 'T','T')
            else:
                if options.debug:
                    info("No transpose")
                a_gpu = gpuarray.to_gpu(a)
                b_gpu = gpuarray.to_gpu(b)
                c_gpu = cu.dot(a_gpu, b_gpu, 'N','N')
                res = c_gpu.get()
    except:
        debug("Falling back on using numpy instead of CUDA")
        res = np.dot(x,y)

    if options.debug:
        mprint("numpy.dot",npdot)
        res = np.ascontiguousarray(res)
        mprint("cu.dot",res)
        assert np.allclose(res,npdot),"GPU does not match numpy"
        print "res.shape",res.shape
        print "npdot.shape",npdot.shape
        print "res.strides",res.strides
        print "npdot.strides",npdot.strides
        assert res.shape == npdot.shape
        assert res.strides == npdot.strides
    debug("exit CUDA dot product - all tests pass")
    return res


