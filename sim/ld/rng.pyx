# rng.pyx
import cython
from libc.stdint cimport uint64_t
from libc.math cimport log, sqrt


cdef class XorShift64Star:
    """
    Simple per-instance RNG with a 64-bit internal state.
    Deterministic and seedable.
    """
    
    def __cinit__(self, uint64_t seed):
        if seed == 0:
            # xorshift64* must not be seeded with 0
            seed = 0xdeadbeefcafebabe
        self.state = seed

    cdef uint64_t next_uint64(self) noexcept nogil:
        """
        Core xorshift64* step.
        Returns next 64-bit unsigned random integer.
        """
        cdef uint64_t x = self.state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.state = x
        return x * 2685821657736338717ULL

    cpdef unsigned long next_uint(self):
        """
        Python-visible: returns a 32-bit-ish integer.
        """
        return <unsigned long>(self.next_uint64())

    cdef double random_c(self) noexcept nogil:
        """
        Random float in [0.0, 1.0).
        """
        cdef uint64_t x = self.next_uint64()
        x >>= 11
        return x * (1.0 / 9007199254740992.0)

    cpdef double random(self):
        """
        Python-visible: random float in [0.0, 1.0).
        """
        return self.random_c()

    cpdef unsigned long randint(self, unsigned long a, unsigned long b):
        """
        Python-visible: randint(a, b), inclusive.
        """
        cdef unsigned long r = self.next_uint()
        return a + r % (b - a + 1)
    
    cdef double random_gaussian_c(self) noexcept nogil:
        cdef double x1, x2, w

        w = 2.0
        while w >= 1.0 or w == 0.0:
            x1 = 2.0 * self.random_c() - 1.0
            x2 = 2.0 * self.random_c() - 1.0
            w = x1 * x1 + x2 * x2

        w = sqrt((-2.0 * log(w)) / w)
        return x1 * w

    cpdef double random_gaussian(self):
        """
        Python-visible wrapper that just calls the nogil C version.
        GIL is held here, which is fine for Python.
        """
        return self.random_gaussian_c()
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void assign_random_gaussian_pair(self, double[:] out, int assign_ix) noexcept nogil:
        """
        Fill out[assign_ix] and out[assign_ix + 1] with N(0,1) samples
        using Box–Muller (Marsaglia polar method) and this RNG.
        """
        cdef double x1, x2, w

        w = 2.0
        while w >= 1.0 or w == 0.0:
            x1 = 2.0 * self.random_c() - 1.0
            x2 = 2.0 * self.random_c() - 1.0
            w = x1 * x1 + x2 * x2

        w = sqrt((-2.0 * log(w)) / w)
        out[assign_ix] = x1 * w
        out[assign_ix + 1] = x2 * w
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void fill_gaussian_c(self, double[:] out) noexcept nogil:
        """
        Fill the whole array `out` with N(0,1) samples.
        Uses pair-generation where possible.
        """
        cdef Py_ssize_t n = out.shape[0]
        cdef Py_ssize_t i

        # handle pairs
        for i in range(0, n - (n % 2), 2):
            self.assign_random_gaussian_pair(out, i)

        # if n is odd, generate one extra
        if n & 1:
            out[n - 1] = self.random_gaussian_c()

    cpdef void fill_gaussian(self, double[:] out):
        """
        Python-visible: fill `out` with N(0,1) samples.
        Accepts any buffer-compatible object (e.g., NumPy array).
        """
        with nogil:
            self.fill_gaussian_c(out)
