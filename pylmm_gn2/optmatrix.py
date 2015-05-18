import sys
import time
import numpy as np
from numpy.distutils.system_info import get_info
from scipy import linalg
from scipy import optimize
from scipy import stats
import cuda

from standalone import uses
import lmmoptions
debug,info,mprint = uses('debug','info','mprint')

initializedMatrix = None
useNumpy = None
useBLAS = False
hasBLAS = None

def matrix_initialize():
    global initializedMatrix
    global useBLAS
    global useNumpy  # module based variable
    global dgemm

    if initializedMatrix:
        sys.stderr.write("INFO: matrix_inialize called multiple times\n")
        return

    if useBLAS and useNumpy is None:
        print get_info('blas_opt')
        try:
            from scipy.linalg.blas import dgemm
            sys.stderr.write("INFO: using linalg.fblas\n")
            useNumpy = False
            hasBLAS  = True
        except AttributeError:
            sys.stderr.write("WARNING: linalg.fblas not found, using numpy.dot instead!\n")
            useNumpy = True
    else:
        sys.stderr.write("INFO: using numpy.dot\n")
        useNumpy=True
    if cuda.useCUDA:
        sys.stderr.write("INFO: with CUDA support\n")
    initializedMatrix = True


def matrixMult(A,B):
   global initializedMartix
   global useNumpy  # module based variables

   options = lmmoptions.get()
   if options.debug:
       debug("enter matrixMult")
       mprint("A",A)
       mprint("B",B)
   if not initializedMatrix:
       matrix_initialize()

   if cuda.useCUDA:
       res = cuda.dot(A,B)
       if options.debug:
           mprint("cuda.dot",res)
       return res

   if useNumpy:
       res = np.dot(A,B)
       if options.debug:
           mprint("np.dot",res)
       return res

   # If the matrices are in Fortran order then the computations will be faster
   # when using dgemm.  Otherwise, the function will copy the matrix and that takes time.
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

   return dgemm(alpha=1.,a=AA,b=BB,trans_a=transA,trans_b=transB)

def matrixMultT(M):
    return matrixMult(M,M.T)
