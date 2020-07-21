
# cython: language_level=2
# distutils: language = c++

from cpython.pycapsule cimport PyCapsule_New

from mpi4py.libmpi cimport MPI_Comm, MPI_FLOAT, MPI_DOUBLE, MPI_Allreduce, MPI_SUM

from libc.stdio cimport printf
from libc.stdint cimport int32_t, int64_t, uint64_t


cdef void sum_inplace_mpi_f32(void* out_ptr, void** data_ptr) nogil:
  cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
  cdef float* x = <float*>(data_ptr[1])
  cdef float* out = <float*>(out_ptr)
  cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[2]))[0])

  MPI_Allreduce(x, out, nitems, MPI_FLOAT, MPI_SUM, comm)

cdef void sum_inplace_mpi_f64(void* out_ptr, void** data_ptr) nogil:
  cdef int32_t nitems = (< int32_t*>(data_ptr[0]))[0]
  cdef double* x = <double*>(data_ptr[1])
  cdef double* out = <double*>(out_ptr)
  cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[2]))[0])
  out[0] = x[0]

  MPI_Allreduce(x, out, nitems, MPI_DOUBLE, MPI_SUM, comm)

cdef void sum_inplace_mpi_c64(void* out_ptr, void** data_ptr) nogil:
  cdef int32_t nitems = (< int32_t*>(data_ptr[0]))[0]
  cdef float complex* x = <float complex*>(data_ptr[1])
  cdef float complex* out = <float complex*>(out_ptr)
  cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[2]))[0])
  out[0] = x[0]

  MPI_Allreduce(x, out, nitems * 2, MPI_FLOAT, MPI_SUM, comm)

cdef void sum_inplace_mpi_c128(void* out_ptr, void** data_ptr) nogil:
  cdef int32_t nitems = (< int32_t*>(data_ptr[0]))[0]
  cdef double complex* x = <double complex*>(data_ptr[1])
  cdef double complex* out = <double complex*>(out_ptr)
  cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[2]))[0])
  out[0] = x[0]

  MPI_Allreduce(x, out, nitems * 2, MPI_DOUBLE, MPI_SUM, comm)

cpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
  cdef const char * name = "xla._CUSTOM_CALL_TARGET"
  cpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)

register_custom_call_target(b"sum_inplace_mpi_f32", < void*>(sum_inplace_mpi_f32))
register_custom_call_target(b"sum_inplace_mpi_f64", < void*>(sum_inplace_mpi_f64))
register_custom_call_target(b"sum_inplace_mpi_c64", < void*>(sum_inplace_mpi_c64))
register_custom_call_target(b"sum_inplace_mpi_c128", < void*>(sum_inplace_mpi_c128))
