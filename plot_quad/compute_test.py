import numpy as np
import ctypes
import pathlib
import time



libname = pathlib.Path().absolute() / "plot_quad/compute.so"
c_lib = ctypes.CDLL(libname)
c_float_p = ctypes.POINTER(ctypes.c_float)
c_float_p_p = np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
c_int_p_p = np.ctypeslib.ndpointer(ctypes.c_int8, flags="C_CONTIGUOUS")
c_float = ctypes.c_float
c_int = ctypes.c_int
byref = ctypes.byref

c_interceptQuad = c_lib.interceptQuad
c_interceptQuad.argtypes = [c_float_p_p, c_float, c_float, c_float, c_float, c_float, c_float, c_float_p, c_float_p, c_float_p]


c_compute3DValues = c_lib.compute3DValues
c_compute3DValues.argtypes = [c_float_p_p, c_int_p_p, c_float, c_int]

c_zeros = c_lib.zeros
c_zeros.argtypes = [c_int]
c_zeros.restype = ctypes.POINTER(ctypes.c_float * 10)


E = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, -1]])

N = 3

pp1x = c_float(0.0)
pp1y = c_float(0.0)
pp1z = c_float(0.0)

accumulatedTime = 0.0

for i in range(N):

    p1x = 10.0*np.random.randn(1).astype(np.float32)
    p1y = 10.0*np.random.randn(1).astype(np.float32)
    p1z = 10.0*np.random.randn(1).astype(np.float32)
    p2x = 10.0*np.random.randn(1).astype(np.float32)
    p2y = 10.0*np.random.randn(1).astype(np.float32)
    p2z = 10.0*np.random.randn(1).astype(np.float32)

    tic = time.time()
    c_interceptQuad(E.astype(np.float32),  p1x, p1y,p1z, p2x, p2y, p2z,   byref(pp1x), byref(pp1y), byref(pp1z))
    toc = time.time()

    accumulatedTime += toc-tic

    print(pp1x, pp1y, pp1z)


lim = 4.0
N = 4

A = np.zeros((N,N,N)).astype(np.int8)

c_compute3DValues(E.astype(np.float32), A, lim, N)

print(A)


print(f'{accumulatedTime}s')


A = c_zeros(10)

print(np.array(A.contents))

