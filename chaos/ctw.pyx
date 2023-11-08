# distutils: language = c++
cpdef estimate_entropy(seq1, alphabet_size):
    return cpp_ctw(seq1, alphabet_size)
