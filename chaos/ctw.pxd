from libcpp.vector cimport vector
cimport cython
cimport numpy as np
import numpy as np

cdef extern from "cppctw.hpp" namespace "ctw":
    cpdef double cpp_ctw(const vector[char] seq, char alphabet_size) except +

