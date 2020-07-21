import ctypes

from jax import core
from jax.interpreters import xla
from jax.lib import xla_client
from jax import abstract_arrays
import numpy as _np
import jax

import mpi4py.MPI as MPI


def sum_inplace_jax(x, comm):
    if not isinstance(x, jax.interpreters.xla.DeviceArray):
        raise TypeError("Argument to sum_inplace_jax must be a DeviceArray, got {}"
                        .format(type(x)))

    _x = jax.xla._force(x.block_until_ready())
    ptr = _x.device_buffer.unsafe_buffer_pointer()

    # rebuild comm
    _comm = MPI.Comm()
    _comm_ptr = ctypes.c_void_p.from_address(MPI._addressof(_comm))
    _comm_ptr.value = int(comm)

    # using native numpy because jax's numpy does not have ctypeslib
    data_pointer = _np.ctypeslib.ndpointer(x.dtype, shape=x.shape)

    # wrap jax data into a standard numpy array which is handled by MPI
    arr = data_pointer(ptr).contents

    _comm.Allreduce(MPI.IN_PLACE, arr, op=MPI.SUM)

    return _x


_ops = xla_client.ops

# The underlying jax primitive
sum_inplace_p = core.Primitive("sum_inplace_mpi")  # Create the primitive


# This function applies the primitive to a AST
def sum_inplace_jax_primitive(x, comm):
    comm_ptr = _np.uint64(MPI._handleof(comm))
    return sum_inplace_p.bind(x, comm=comm_ptr)


# This function executes the primitive, when not under any transformation
sum_inplace_p.def_impl(sum_inplace_jax)


# This function evaluates only the shapes during AST construction
def sum_inplace_abstract_eval(xs, comm):
    return abstract_arrays.ShapedArray(xs.shape, xs.dtype)


sum_inplace_p.def_abstract_eval(sum_inplace_abstract_eval)


# Helper functions

def _constant_s32_scalar(c, x):
    return _ops.Constant(c, _np.int32(x))


def _unpack_builder(c):
    # If `c` is a ComputationBuilder object, extracts the underlying XlaBuilder.
    return getattr(c, "_builder", c)


#  This function compiles the operation
def sum_inplace_xla_encode(c, x, comm):
    c = _unpack_builder(c)
    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = dims[0]
    for el in dims[1:]:
        nitems *= el

    # those kernels have been loaded through cython.
    if dtype == _np.float32:
        kernel = b"sum_inplace_mpi_f32"
    elif dtype == _np.float64:
        kernel = b"sum_inplace_mpi_f64"
    elif dtype == _np.complex64:
        kernel = b"sum_inplace_mpi_c64"
    elif dtype == _np.complex128:
        kernel = b"sum_inplace_mpi_c128"

    return _ops.CustomCall(
        c,
        kernel,
        operands=(
            xla_client.ops.Constant(c, _np.int32(nitems)),
            x,
            xla_client.ops.Constant(c, comm),
        ),
        shape=xla_client.Shape.array_shape(dtype, dims),
    )


# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][sum_inplace_p] = sum_inplace_xla_encode
