# rng.pxd
from libc.stdint cimport uint64_t

cdef class XorShift64Star:
    cdef uint64_t state

    # Core C-level, no-GIL API
    cdef uint64_t next_uint64(self) noexcept nogil
    cdef double random_c(self) noexcept nogil
    cdef double random_gaussian_c(self) noexcept nogil
    cdef void assign_random_gaussian_pair(self, double[:] out, int assign_ix) noexcept nogil
    cdef void fill_gaussian_c(self, double[:] out) noexcept nogil

    # Python-visible API
    cpdef unsigned long next_uint(self)
    cpdef double random(self)
    cpdef double random_gaussian(self)
    cpdef unsigned long randint(self, unsigned long a, unsigned long b)
    cpdef void fill_gaussian(self, double[:] out)
